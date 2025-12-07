# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
Modified to support generating mixed validation sets (0% to 100%)
"""

import os
import datasets
import argparse
import numpy as np
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    # 新增参数：是否生成混合比例的数据集
    parser.add_argument('--generate_mixed', action='store_true', help='Generate datasets with 0.0 to 1.0 validation set ratios')

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=False)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                },
                # 临时存储完整的回复内容，后续根据比例决定是否放入 response
                "_full_response": [{
                    "role": "assistant",
                    "content": answer
                }]
            }
            return data

        return process_fn

    print("Processing datasets...", flush=True)
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # 转换为 Pandas DataFrame 以便进行灵活的比例控制
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    # 测试集通常不需要 response（用于生成），且保持固定
    if '_full_response' in test_df.columns:
        test_df = test_df.drop(columns=['_full_response'])
    test_df['response'] = None # 确保列存在

    def save_dataset(df, ratio, base_dir, filename='train.parquet'):
        """
        根据 ratio 比例，将 _full_response 移动到 response 字段
        """
        df_out = df.copy()
        
        # 初始化 response 列为空
        df_out['response'] = None
        
        if ratio > 0.0:
            # 设定随机种子以保证 10% 是 20% 的子集 (Consistency)
            np.random.seed(42)
            # 生成掩码：约 ratio 比例的行为 True
            mask = np.random.rand(len(df_out)) < ratio
            
            # 将选定行的完整答案赋值给 response
            df_out.loc[mask, 'response'] = df_out.loc[mask, '_full_response']
            count = mask.sum()
            print(f"  Ratio {ratio}: {count}/{len(df_out)} samples contain ground truth response.")
        else:
            print(f"  Ratio {ratio}: 0 samples contain ground truth response.")

        # 删除临时列
        if '_full_response' in df_out.columns:
            df_out = df_out.drop(columns=['_full_response'])
            
        # 确定保存路径
        if args.generate_mixed:
            # 例如: data/math/p_0.1/train.parquet
            out_dir = os.path.join(base_dir, f'p_{ratio:.1f}')
        else:
            out_dir = base_dir
            
        makedirs(out_dir)
        final_path = os.path.join(out_dir, filename)
        df_out.to_parquet(final_path)
        print(f"Saved to {final_path}")

    local_dir = args.local_dir
    
    # 执行生成逻辑
    if args.generate_mixed:
        print("Generating mixed datasets from 0% to 100%...", flush=True)
        # 生成 0.0, 0.1, ... 1.0
        ratios = [round(x * 0.1, 1) for x in range(11)]
        for r in ratios:
            save_dataset(train_df, r, local_dir)
        
        # 保存一份公共的 test.parquet 到根目录 (或者每个子目录都复制一份亦可)
        test_path = os.path.join(local_dir, 'test.parquet')
        test_df.to_parquet(test_path)
        print(f"Saved test dataset to {test_path}")

    else:
        # 默认行为：0% (纯 Online，不带 response)
        print("Generating default online dataset (0% mix)...", flush=True)
        save_dataset(train_df, 0.0, local_dir)
        test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # HDFS 上传逻辑
    if args.hdfs_dir is not None:
        print(f"Copying data to HDFS: {args.hdfs_dir}", flush=True)
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)