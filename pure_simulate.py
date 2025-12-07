import numpy as np


def debug_reward_weighting_with_backward_min(case_name, score_list, temperature=0.1):
    print(f"\n{'=' * 20} {case_name} {'=' * 20}")

    # 1. 准备数据
    rm_score = np.array(score_list, dtype=np.float64)
    reward_mask = (rm_score != 0)  # 假设非0位置是有效的Step

    print(f"原始分数:\n{rm_score}")

    # ==========================================
    #  新增逻辑：从后往前，传播最小值
    # ==========================================

    # 获取所有有效步骤的索引 (e.g., [2, 5, 8...])
    step_indices = np.where(reward_mask)[0]

    # 只有当至少有2个步骤时才需要传播
    if len(step_indices) > 1:
        # 这里的逻辑是：Current_Step = Min(Current_Step, Future_Steps_Min)
        # 我们从倒数第二个有效步骤开始，向前遍历

        # 初始化 running_min 为最后一个步骤的分数
        running_min = rm_score[step_indices[-1]]

        # 从倒数第二个索引开始，步长为-1，直到第0个索引
        for i in range(len(step_indices) - 2, -1, -1):
            curr_idx = step_indices[i]
            curr_val = rm_score[curr_idx]

            # 核心计算：当前值 vs 后续的最小值
            # 如果后续有更低的分数，当前分数会被拉低；如果后续分数很高，当前分数保持不变
            new_val = min(curr_val, running_min)

            rm_score[curr_idx] = new_val

            # 更新 running_min，供更前面的步骤使用
            running_min = new_val

    print(f"\n[Step New] 向后最小值传播后的分数:\n{rm_score}")
    print("(注意：前面的高分被后面的低分'拉'下来了，形成单调非递增序列)")

    # ==========================================
    #  原有的加权逻辑 (基于处理后的分数)
    # ==========================================

    # 1. Masked Fill (无效位变 inf)
    filled_score = rm_score.copy()
    # 注意：这里要用原来的 mask，因为位置没变，只是值变了
    filled_score[~reward_mask] = np.inf

    # 2. 负号与温度 (Softmax前置处理)
    logits = -filled_score / temperature

    # 3. Softmax
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    weight = exp_logits / np.sum(exp_logits)

    print(f"\n[Step 3] 计算权重 (基于新分数的 Softmax):\n{np.round(weight, 4)}")

    # 4. 最终加权
    final_score = rm_score * weight
    print(f"\n[Step 4] 最终结果 (New Score * Weight):\n{np.round(final_score, 4)}")
    print(f"权重总和: {np.sum(weight):.4f}")


# --- 运行测试用例 ---

# Case 1: “晚节不保”型
# 前面是 3分、2分（很高），但最后一步只有 1分。
# 预期：前面的 3 和 2 都会被最后那个 1 拉下来，变成 [1, 1, 1]。
case1_scores = [0, 0, 3, 0, 0, 2, 0, 0, 1]
debug_reward_weighting_with_backward_min("Case 1 (晚节不保)", case1_scores)

# Case 2: “渐入佳境”型
# 前面是 1分，后面变成了 3分。
# 预期：前面的 1 保持不变（因为 min(1, 3) = 1），不会被后面拉高。
case2_scores = [0, 0, 1, 0, 0, 2, 0, 0, 3]
debug_reward_weighting_with_backward_min("Case 2 (渐入佳境)", case2_scores)

# Case 3: “中间拉胯”型
# [3, 1, 3] -> 中间的 1 会把前面的 3 拉下来，但后面的 3 不受影响。
# 预期结果： [1, 1, 3]
case3_scores = [0, 0, 3, 0, 0, 1, 0, 0, 3]
debug_reward_weighting_with_backward_min("Case 3 (中间拉胯)", case3_scores)