#清洗训练数据

DATAPATH = "stanford-rna-3d-folding/"
TRAIN_SEQ_FILE_PATH = f"{DATAPATH}/train_sequences.v2.csv"
TRAIN_LABEL_FILE_PATH = f"{DATAPATH}/train_labels.v2.csv"
VALI_SEQ_FILE_PATH = f"{DATAPATH}/validation_sequences.csv"
VALI_LABEL_FILE_PATH = f"{DATAPATH}/validation_labels.csv"
TEST_SEQ_FILE_PATH = f"{DATAPATH}/test_sequences.csv"

TRAIN_LABEL_FILE_CLEANED_PATH = f"2_cleaned.csv"
TRAIN_SEQ_FILE_CLEANED_PATH = f"1_cleaned.csv"



import pandas as pd

def process_csv_files(a_file, b_file):
    """
    处理两个 CSV 文件，根据 b 文件中的空值删除相关行。

    Args:
        a_file (str): a.csv 文件的路径。
        b_file (str): b.csv 文件的路径。

    Returns:
        tuple: 包含处理后的 DataFrame (df_a, df_b)。
    """
    try:
        df_a = pd.read_csv(a_file)
        df_b = pd.read_csv(b_file)
    except FileNotFoundError:
        print("错误：指定的文件未找到，请检查文件路径是否正确。")
        return None, None

    # 查找 b 文件中 x_1, y_1, z_1 列存在空值的行
    null_rows_b = df_b[df_b[['x_1', 'y_1', 'z_1']].isnull().any(axis=1)]

    # 提取需要删除的 ID 前缀
    # prefixes_to_remove = set(row['ID'].split('_')[0] for index, row in null_rows_b.iterrows())
    prefixes_to_remove = set(extract_prefix(row) for index, row in null_rows_b.iterrows())


    # 删除 b 文件中所有包含这些前缀的行
    df_b_cleaned = df_b[~df_b['ID'].str.startswith(tuple(prefixes_to_remove))]

    # 删除 a 文件中 target_id 包含这些前缀的行
    df_a_cleaned = df_a[~df_a['target_id'].str.startswith(tuple(prefixes_to_remove))]

    return df_a_cleaned, df_b_cleaned


#比较清洗后的数据数目
def compare_file(str,file_path1,file_path2):
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
    except FileNotFoundError:
        print("错误：找不到指定的文件。请检查文件路径是否正确。")
        return
    len1 = len(df1)
    len2 = len(df2)
    print(str,len1,len2,len1-len2)
    
# 定义一个函数来提取 "_数字" 前的字符串
def extract_prefix(text):
    if isinstance(text, str):
        last_underscore_index = text.rfind("_")
        if last_underscore_index != -1:
            return text[:last_underscore_index]
    return None

def file_is_null():
    df_b = pd.read_csv(TRAIN_LABEL_FILE_CLEANED_PATH)
    null_rows_b = df_b[df_b[['x_1', 'y_1', 'z_1']].isnull().any(axis=1)]
    print("null_rows: ",null_rows_b)


def verify_data(a_file, b_file):
    """
    验证 a.csv 的 target_id 是否包含在 b.csv 的 ID 前缀中，反之亦然。

    Args:
        a_file (str): a.csv 文件的路径。
        b_file (str): b.csv 文件的路径。
    """
    try:
        df_a = pd.read_csv(a_file)
        df_b = pd.read_csv(b_file)
    except FileNotFoundError:
        print("错误：指定的文件未找到，请检查文件路径是否正确。")
        return

    # 获取 a.csv 中所有的 target_id
    target_ids_a = set(df_a['target_id'])

    # 获取 b.csv 中所有 ID 的前缀
    # prefixes_b = set(id_val.split('_')[0] for id_val in df_b['ID'])
    prefixes_b = set(extract_prefix(id_val) for id_val in df_b['ID'])
    # print("prefixes_b: ",prefixes_b)


    # 检查 a 中的 target_id 是否都包含在 b 的 ID 前缀中
    not_in_b = target_ids_a - prefixes_b
    if not not_in_b:
        print("验证通过：a.csv 中的所有 target_id 都包含在 b.csv 的 ID 前缀中。")
    else:
        print("验证失败：以下 target_id 在 b.csv 的 ID 前缀中找不到：")
        print(not_in_b)

    print("-" * 30)

    # 检查 b 的 ID 前缀是否都包含在 a 的 target_id 中
    not_in_a = prefixes_b - target_ids_a
    if not not_in_a:
        print("验证通过：b.csv 的所有 ID 前缀都包含在 a.csv 的 target_id 中。")
    else:
        print("验证失败：以下 b.csv 的 ID 前缀在 a.csv 的 target_id 中找不到：")
        print(not_in_a)






if __name__ == "__main__":
    file2 = TRAIN_LABEL_FILE_PATH  # 请替换为你的第一个 CSV 文件路径
    file1 = TRAIN_SEQ_FILE_PATH  # 请替换为你的第二个 CSV 文件路径
    file4 = TRAIN_LABEL_FILE_CLEANED_PATH
    file3 = TRAIN_SEQ_FILE_CLEANED_PATH

# # 调用函数进行处理
#     df_a_processed, df_b_processed = process_csv_files(file1, file2)

# # 打印处理后的 DataFrame (你可以选择将它们保存到新的 CSV 文件)
#     if df_a_processed is not None and df_b_processed is not None:
#         print("处理后的 a.csv:")
#         print(df_a_processed)
#         print("\n处理后的 b.csv:")
#         print(df_b_processed)

#     # 如果需要保存到新的 CSV 文件，取消注释下面的代码
#     df_a_processed.to_csv('1_cleaned.csv', index=False)
#     df_b_processed.to_csv('2_cleaned.csv', index=False)


    # compare_file("训练序列对比： ",TRAIN_SEQ_FILE_PATH,TRAIN_SEQ_FILE_CLEANED_PATH)
    # compare_file("训练label对比： ",TRAIN_LABEL_FILE_PATH,TRAIN_LABEL_FILE_CLEANED_PATH)
    # ''' 训练序列对比：  5135 2578 2557
    #     训练label对比：  3677095 2419595 1257500'''
    # file_is_null()
    
    
    
    # 指定你的文件名 (确保使用你处理后的文件，例如 'a_cleaned.csv' 和 'b_cleaned.csv')
    file_a_verified = '1_cleaned.csv'  # 如果你保存了处理后的文件，请使用新文件名
    file_b_verified = '2_cleaned.csv'  # 如果你保存了处理后的文件，请使用新文件名

    #调用验证函数
    # verify_data(file_a_verified, file_b_verified)
    file_is_null()

