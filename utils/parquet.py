import os
import pandas as pd
import argparse

import re

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_solution_math(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def normalize_columns(df, data_source=None):
    # 定义可接受字段（按优先级）
    aliases = {
        "question": ["question", "input", "problem", "prompt", "query", "content", "Question"],
        "answer": ["answer", "output", "response", "result", "correct", "target", "final_answer", "Answer", "solution"]
    }

    normalized = {}

    for target_col, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                if target_col == "answer":
                    if data_source == "minerva_math":
                        print("[INFO]: Extracting math solution from column: ")
                        normalized[target_col] = df[c].apply(extract_solution_math)
                    elif data_source == "gsm8k":
                        print("[INFO]: Extracting gsm8k solution from column: ")
                        normalized[target_col] = df[c].apply(extract_solution_gsm8k)
                    else:
                        normalized[target_col] = df[c]
                    break
                else:
                    normalized[target_col] = df[c]  # 取第一匹配
                    break
                
        else :
            # 如果完全没有该字段，填None占位（可选）
            normalized[target_col] = None 
            print(f"[Warning]: column {target_col} not found in {df.columns}")

    return pd.DataFrame(normalized)

def convert_jsonl_to_parquet(jsonl_path, output_path):
    """Convert a jsonl file to a parquet file."""
    try: 
        df = pd.read_json(jsonl_path, lines=True)
        data_source = os.path.basename(os.path.dirname(jsonl_path))
        df = normalize_columns(df, data_source)
        df["data_source"] = data_source
        df.to_parquet(output_path, index=False)
        print(f"✔ Converted: {jsonl_path} → {output_path}")
    except Exception as e:
        print(f"❌ Error converting {jsonl_path}: {e}")


def process_folder(root_dir, output_dir):
    """Traverse folders and convert test.jsonl in each subfolder."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "test.jsonl" in filenames:
            jsonl_file = os.path.join(dirpath, "test.jsonl")
            folder_name = os.path.basename(dirpath)
            # output_file = os.path.join(dirpath, f"{folder_name}.parquet")
            output_file = os.path.join(output_dir, f"{folder_name}.parquet")
            convert_jsonl_to_parquet(jsonl_file, output_file)

def check_parquet(file_path):
    df = pd.read_parquet(file_path)
    print("[PREVIEW]: \n", df.head())   # 查看前几行
    # print("[COLUMNS]: \n", df.columns)  # 查询列
    # print("[INFO]: \n", df.info())   # 数据结构

def check_pqrquet_dir(dir_path):
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".parquet"):
            check_parquet(os.path.join(dir_path, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert test.jsonl files in subfolders to parquet")
    parser.add_argument("--root_path", "-r", required=True, help="Root folder path")
    parser.add_argument("--output_path", "-o", required=True)

    args = parser.parse_args()
    process_folder(args.root_path, args.output_path)
    
    # check_parquet("D:/Code/Projects/PURE/utils/parquet/aqua.parquet")
    # check_parquet("D:/Code/Projects/PURE/aime2024.parquet")
    # check_parquet("D:\Code\Projects\PURE\math500.parquet")
    # check_parquet("D:\Code\Projects\PURE\\train.parquet")
    
    check_pqrquet_dir("D:\\Code\\Projects\\PURE\\utils\\parquet")    
    