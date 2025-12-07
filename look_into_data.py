import pandas as pd

# df = pd.read_parquet("math500.parquet")
# df = pd.read_parquet("aime2024.parquet")
# df = pd.read_parquet("train3_5.parquet")
# df = pd.read_parquet("train_gsm8k.parquet")
df = pd.read_parquet("train.parquet")
print(df.shape) 
print(df.head(56878))  # 查看前几行
# print(df["answer"].isna().sum())