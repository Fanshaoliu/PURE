import pynvml
import time
import subprocess
import argparse
import os
import sys

def initialize_nvml():
    """初始化NVIDIA管理库"""
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"初始化NVML失败: {e}")
        return False

def get_gpu_count():
    """返回系统GPU数量"""
    return pynvml.nvmlDeviceGetCount()

def parse_gpu_indices(gpus_str, device_count):
    """
    将形如 '0,1,2,3' 的字符串解析为索引列表，并做边界检查与去重排序
    """
    if gpus_str is None or gpus_str.strip() == "":
        # 未指定则默认监控全部
        return list(range(device_count))

    parts = [s.strip() for s in gpus_str.split(",") if s.strip() != ""]
    indices = []
    for p in parts:
        if not p.isdigit() and not (p.startswith("-") and p[1:].isdigit()):
            raise ValueError(f"非法GPU索引: {p}")
        idx = int(p)
        if idx < 0 or idx >= device_count:
            raise ValueError(f"GPU索引越界: {idx} (共有 {device_count} 张)")
        indices.append(idx)
    # 去重并按物理序排序（保持更直观）
    indices = sorted(set(indices))
    return indices

def is_gpu_idle(gpu_index, utilization_threshold=5, memory_threshold=5):
    """
    检查指定GPU是否处于空闲状态
    utilization_threshold: GPU利用率阈值(%)，低于此值视为空闲
    memory_threshold: 显存利用率阈值(%)，低于此值视为空闲
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    # 获取GPU利用率信息
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu

    # 获取显存使用信息
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_util = (mem_info.used / mem_info.total) * 100

    # 打印GPU状态（可选）
    print(f"GPU {gpu_index}: 利用率={gpu_util}%，显存使用率={mem_util:.1f}% ", end="| ")

    return (gpu_util < utilization_threshold) and (mem_util < memory_threshold)

def selected_gpus_idle(gpu_indices, util_threshold=5, mem_threshold=5):
    """仅检查所选GPU是否都处于空闲状态"""
    print(f"\n[{time.ctime()}] 检查所选GPU状态: {gpu_indices}")
    all_idle = True
    busy_list = []
    for i in gpu_indices:
        if not is_gpu_idle(i, util_threshold, mem_threshold):
            all_idle = False
            busy_list.append(i)
    if not all_idle:
        print(f"\n繁忙GPU: {busy_list}")
    return all_idle

def launch_program(command, visible_devices):
    """启动指定程序，并设置CUDA_VISIBLE_DEVICES为所选GPU"""
    vis_str = ",".join(str(i) for i in visible_devices)
    print(f"\n[{time.ctime()}] 所选GPU均空闲，设置 CUDA_VISIBLE_DEVICES={vis_str} 启动程序...")
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = vis_str
        subprocess.Popen(command, shell=True, env=env)
        print(f"程序已启动: {command}")
        return True
    except Exception as e:
        print(f"启动程序失败: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPU监控与程序启动器（支持指定GPU索引）')
    parser.add_argument('--command', required=True, help='当所选GPU空闲时要执行的命令')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔时间(秒)，默认60秒')
    parser.add_argument('--util-threshold', type=int, default=5, help='GPU利用率空闲阈值(%)，默认5%')
    parser.add_argument('--mem-threshold', type=int, default=5, help='显存利用率空闲阈值(%)，默认5%')
    parser.add_argument('--gpus', type=str, default=None, help='要排队的GPU索引，形如"0,1,2,3"；缺省为全部GPU')

    args = parser.parse_args()

    # 初始化NVML
    if not initialize_nvml():
        sys.exit(1)

    try:
        device_count = get_gpu_count()
        try:
            gpu_indices = parse_gpu_indices(args.gpus, device_count)
        except ValueError as ve:
            print(f"GPU索引解析错误: {ve}")
            sys.exit(1)

        print(f"检测到 {device_count} 张GPU；将监控这些GPU: {gpu_indices}")
        print(f"空闲阈值: 利用率<{args.util_threshold}% 且 显存<{args.mem_threshold}%")
        print(f"检查间隔: {args.interval} 秒")
        print("开始监控... 按 Ctrl+C 结束。")

        while True:
            if selected_gpus_idle(gpu_indices, args.util_threshold, args.mem_threshold):
                if launch_program(args.command, gpu_indices):
                    break
            else:
                print(f"空闲阈值: 利用率<{args.util_threshold}% 且 显存<{args.mem_threshold}%")
                print(f"[{time.ctime()}] 所选GPU未全部空闲，{args.interval} 秒后再次检查...")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()


'''
# 监控 0,1,2,3 四张卡，阈值都用 5%，每 10 秒检查一次，空闲即启动脚本
python launch_shell.py \
  --gpus "0,1,2,3,4,5,6,7" \
  --command "bash examples/scripts_test/test_rm_llama.sh" \
  --interval 1 \
  --util-threshold 5 \
  --mem-threshold 5
'''