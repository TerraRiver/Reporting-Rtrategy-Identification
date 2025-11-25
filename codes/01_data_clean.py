import pandas as pd

# 0. 全局设置
pd.set_option('display.max_colwidth', None) # 显示完整列宽

# 1. 设置常量
Raw_Data_Path = 'data/raw/articles.json'

# 2. 功能函数
def load_raw_data(file_path):
    """加载原始数据"""
    print("-" * 50) # 打印分隔线
    print(f"正在加载原始数据{file_path}")
    df = pd.read_json(file_path)
    df = df['articles'].apply(pd.Series)  # 展开嵌套的articles字段
    print(f"数据加载完成，共加载了{len(df)}篇文章，列名如下：")
    print(df.columns)
    print("-" * 50) # 打印分隔线
    return df

def basic_clean(df):
    """基础清理函数"""
    print("-" * 50) # 打印分隔线
    # 规范化日期结构
    print("正在规范化时间格式")
    print(f"转换前的publication_time示例: {df.iloc[5]['publication_date']}")
    df['publish_date'] = pd.to_datetime(df['publication_date'])
    print(f"转换后的publish_date示例: {df.iloc[5]['publish_date']}")
    # 删除不需要的列
    print("正在删除不需要数据列")
    df = df.drop(columns=['publication_time','publication_date','author'])
    print(f"剩余列名如下: {df.columns.tolist()}")
    # 重命名列
    print("正在重命名列")
    df.rename(columns={
    'headline': 'title',
    'source': 'source_media'
    }, inplace=True)
    print(f"重命名后列名如下: {df.columns.tolist()}")
    # 去除重复数据（重复判定规则为标题+内容）
    num_1 = len(df)
    print("正在去除重复数据，去除前数据量为:", num_1)
    df.drop_duplicates(subset=['title', 'content'], inplace=True)
    num_2 = len(df)
    print("共去除重复数据:", num_1 - num_2, "去除后数据量为:", num_2)
    print("-" * 50) # 打印分隔线
    return df

def meida_clean(df):
    """清洗媒体来源函数"""


# 3. 主流程控制函数
def main():
    # 加载原始数据
    raw_data = load_raw_data(Raw_Data_Path)
    # 基础清理
    basic_cleaned_data = basic_clean(raw_data)

if __name__ == "__main__":
    main()