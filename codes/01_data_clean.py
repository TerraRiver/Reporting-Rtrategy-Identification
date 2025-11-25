import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import platform

# 0. 全局设置
# 打印设置
pd.set_option('display.max_colwidth', None) # 显示完整列宽
# 设置 Matplotlib 和 Seaborn 样式
# 重置默认配置
plt.rcdefaults()
# 设置 Seaborn 主题为 "ticks" (白色背景，有刻度)，上下文为 "paper" (适合论文的大小)
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
# 判断操作系统以选择合适的衬线中文字体
system_name = platform.system()
if system_name == "Windows":
    font_list = ['Times New Roman', 'SimSun', 'STSong'] # Windows: Times + 宋体
elif system_name == "Darwin":
    font_list = ['Times New Roman', 'Songti SC', 'STSong'] # Mac: Times + 宋体
else:
    font_list = ['Times New Roman', 'Noto Serif CJK SC', 'WenQuanYi Zen Hei'] # Linux
# Matplotlib 深度定制
plt.rcParams.update({
    'font.family': 'serif',          # 强制使用衬线体
    'font.serif': font_list,         # 设定衬线字体优先列表
    'axes.unicode_minus': False,     # 解决负号显示问题
    'mathtext.fontset': 'stix',      # 数学公式使用类 LaTeX 字体
    'figure.figsize': (10, 6),       # 默认图表大小
    'axes.labelsize': 12,            # 轴标签字号
    'xtick.labelsize': 10,           # X轴刻度字号
    'ytick.labelsize': 10,           # Y轴刻度字号
    'axes.linewidth': 1.0,           # 坐标轴线宽
    'grid.linestyle': '--',          # 网格线样式（如果开启）
    'grid.alpha': 0.3,               # 网格线透明度
    'savefig.dpi': 300,              # 保存分辨率
    'savefig.bbox': 'tight'          # 保存时自动裁剪空白
})
print(f"完成可视化设置，当前系统: {system_name}，字体配置优先顺序: {font_list[1]}")


# 1. 设置常量
RAW_DATA_PATH = Path("data/raw/articles.json") # 原始数据路径
PLT_RESULT_PATH = Path("results/charts") # 图表结果保存路径

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
    print("-" * 50) # 打印分隔线
    # 打印所有唯一的媒体来源及其文章数量
    source_counts = df['source_media'].value_counts()
    print("\n合并前媒体来源分布情况:")
    print(source_counts)
    # 媒体来源合并与清洗
    print("\n--- 正在进行媒体来源合并与清洗 ---")
    # 合并 'The Times of India' 的相关媒体来源
    df.loc[df['source_media'].str.contains('The Times of India', case=False, na=False), 'source_media'] = 'The Times of India'
    # 合并 'Indian Express' 的相关媒体来源
    df.loc[df['source_media'].str.contains('Indian Express', case=False, na=False), 'source_media'] = 'Indian Express'
    # 合并 'The Economic Times' 的相关媒体来源
    df.loc[df['source_media'].str.contains('The Economic Times', case=False, na=False), 'source_media'] = 'The Economic Times'
    # 合并 'The Hindu' 的相关媒体来源
    df.loc[df['source_media'].str.contains('The Hindu', case=False, na=False), 'source_media'] = 'The Hindu'
    # 删除指定的媒体来源
    sources_to_remove = ['HT Columnists', 'ET Now', 'Mirror Now']
    print(f"正在删除以下来源的新闻: {', '.join(sources_to_remove)}")
    initial_rows = len(df)
    df = df[~df['source_media'].isin(sources_to_remove)]
    print(f"删除了 {initial_rows - len(df)} 篇文章。")
    # 打印合并筛选后的媒体来源分布
    source_counts_final = df['source_media'].value_counts()
    print("\n合并筛选后的媒体来源分布:")
    print(source_counts_final)
    # 可视化合并后的媒体来源分布
    print("正在生成合并筛选后媒体来源分布图")
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6)) # 显式创建对象以便更好地控制
    # 绘制条形图
    # palette 使用 "mako" (深蓝绿色系) 或 "gray" (灰度) 更符合黑白印刷或学术风格
    # zorder=3 让柱状图浮在网格线之上
    sns.barplot(
        x=source_counts_final.values, 
        y=source_counts_final.index, 
        palette="mako", 
        ax=ax,
        edgecolor="black", # 给柱子加黑边，增强对比度
        linewidth=0.8,
        zorder=3
    )
    # 标题和标签 (使用 Times New Roman 风格的字体)
    ax.set_title('媒体来源分布图', fontweight='bold', pad=20)
    ax.set_xlabel('文章数量')
    ax.set_ylabel('媒体来源')
    # 学术图表关键调整：去边框 (Despine)
    sns.despine(trim=True) # 去掉上方和右侧边框，trim=True 让坐标轴线只延伸到数据范围内
    # 仅在 X 轴添加轻微的网格线辅助读数
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    # 保存图片
    save_path = PLT_RESULT_PATH / '媒体来源分布图.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("媒体来源分布图已保存至:", PLT_RESULT_PATH / '媒体来源分布图.png')
    print("-" * 50) # 打印分隔线
    return df

# 3. 主流程控制函数
def main():
    # 加载原始数据
    raw_data = load_raw_data(RAW_DATA_PATH)
    # 基础清理
    basic_cleaned_data = basic_clean(raw_data)
    # 媒体来源清理
    media_cleaned_data = meida_clean(basic_cleaned_data)

if __name__ == "__main__":
    main()
