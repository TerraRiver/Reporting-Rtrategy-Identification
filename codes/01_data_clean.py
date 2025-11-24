import pandas as pd

# 0. 全局设置
pd.set_option('display.max_colwidth', None) # 显示完整列宽

# 1. 设置常量
Raw_Data_Path = 'data/raw/articles.json'


# 2. 功能函数
def load_raw_data(file_path):
    """加载原始数据"""
    print(f"正在加载原始数据{file_path}")
    df = pd.read_json(file_path)
    df = df['articles'].apply(pd.Series)  # 展开嵌套的articles字段
    print("-" * 50) # 打印分隔线
    print(f"数据加载完成，共加载了{len(df)}篇文章，列名如下：")
    print(df.columns)
    print("-" * 50) # 打印分隔线
    return df

def load_raw_data(file_path):



# 3. 主流程控制函数
def main():
    # 加载原始数据
    raw_data = load_raw_data(Raw_Data_Path)
    # 这里可以添加更多的数据清洗步骤
    df_new_test = raw_data.iloc[5]
    print(df_new_test['id'])
    print(df_new_test['publication_date'])
    print(df_new_test['publication_time'])
    df_new_test['publish_date'] = pd.to_datetime(df_new_test['publication_date'] + ' ' + df_new_test['publication_time'], errors='coerce')
    print(df_new_test['publish_date'])

if __name__ == "__main__":
    main()