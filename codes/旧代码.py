# 导入依赖库
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
import spacy
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# 设置 Matplotlib 和 Seaborn 样式
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 100)
print("完成可视化设置")

# 定义输出目录
RESULT_DIR = Path("results/data_prepare")
PROCESSED_DATA_DIR = Path("data/processed")
# 创建输出目录，如果不存在
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
print("完成输出目录设置")

# --- 1. 数据加载与初步处理 ---
# 加载原始数据
data_file = Path("data/raw/articles.json")
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data['articles'])
print(f"成功从 {data_file} 加载了 {len(df)} 篇文章。")

print("正在规范化时间格式")
# 合并日期和时间，并转换为datetime对象
df['publish_date'] = pd.to_datetime(df['publication_date'] + ' ' + df['publication_time'], errors='coerce')

print("正在筛选所需数据列")
# 重命名列以匹配后续流程的期望名称
df.rename(columns={
    'headline': 'title',
    'source': 'source_media'
}, inplace=True)

# 选择并排序所需的列，丢弃不再需要的原始列
required_columns = ['title', 'source_media', 'publish_date', 'content']
df = df[required_columns]

# --- 2. 数据清洗 ---
print("正在进行数据清洗")
len_before = len(df)
print(f"\n清洗前的数据量: {len_before}")

# 检查 'content' 和 'publish_date' 列的缺失值 (errors='coerce' 会将格式错误的日期转为NaT)
df.dropna(subset=['content', 'publish_date'], inplace=True)
print(f"移除内容或日期缺失的行后: {len(df)}")

# 移除完全重复的文章
df.drop_duplicates(subset=['title', 'content'], inplace=True)
print(f"移除重复项后: {len(df)}")

# --- 3. 媒体来源分析与整合 ---
# 打印所有唯一的媒体来源及其文章数量
source_counts = df['source_media'].value_counts()
print("\n合并前媒体来源分布情况:")
print(source_counts)

# 可视化媒体来源分布
plt.figure(figsize=(12, 8))
sns.barplot(x=source_counts.values, y=source_counts.index, palette="viridis")
plt.title('各媒体来源的文章数量分布（合并前）', fontsize=16)
plt.xlabel('文章数量', fontsize=12)
plt.ylabel('媒体来源', fontsize=12)
plt.tight_layout()
plt.savefig(RESULT_DIR / '媒体来源分布图（合并筛选前）.png', dpi=300)

# 媒体来源合并与清洗
print("\n--- 正在进行媒体来源合并与清洗 ---")
# 1. 合并 'The Times of India' 的相关媒体来源
df.loc[df['source_media'].str.contains('The Times of India', case=False, na=False), 'source_media'] = 'The Times of India'
# 2. 合并 'Indian Express' 的相关媒体来源
df.loc[df['source_media'].str.contains('Indian Express', case=False, na=False), 'source_media'] = 'Indian Express'
# 3. 合并 'The Economic Times' 的相关媒体来源
df.loc[df['source_media'].str.contains('The Economic Times', case=False, na=False), 'source_media'] = 'The Economic Times'
# 4. 合并 'The Hindu' 的相关媒体来源
df.loc[df['source_media'].str.contains('The Hindu', case=False, na=False), 'source_media'] = 'The Hindu'

# 5. 删除指定的媒体来源
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
plt.figure(figsize=(12, 8))
sns.barplot(x=source_counts_final.values, y=source_counts_final.index, palette="viridis")
plt.title('合并筛选后媒体来源的文章数量分布', fontsize=16)
plt.xlabel('文章数量', fontsize=12)
plt.ylabel('媒体来源', fontsize=12)
plt.tight_layout()
plt.savefig(RESULT_DIR / '媒体来源分布图（合并筛选后）.png', dpi=300)


# --- 4. 文章时间分布可视化 ---

# 按月对文章进行分组计数
monthly_counts = df.set_index('publish_date').resample('M').size()

# 可视化时间分布
plt.figure(figsize=(15, 6))
monthly_counts.plot(kind='line', marker='o', linestyle='-')
plt.title('每月文章数量时间序列', fontsize=16)
plt.xlabel('月份', fontsize=12)
plt.ylabel('文章数量', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(RESULT_DIR / '文章时间分布图.png', dpi=300)


# --- 5. 文本深度预处理 ---

def basic_clean(text):
    """基础清洗函数，用于语义分析。"""
    # 注意：相关的库 re 和 BeautifulSoup 已在脚本顶部导入
    if not isinstance(text, str):
        return ""
    # 移除HTML标签
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 转为小写，移除多余空格
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text
print("\n预处理函数定义完成。")

# 加载深度清洗模型
print("正在加载和配置spaCy模型...")
# 1. 加载Spacy模型，仅保留tagger用于词形还原(lemmatization)，禁用其他组件以提高效率
nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

# 2. 自定义停用词列表并添加到spaCy词汇表中
custom_stopwords = [
    'india', 'china', 'indian', 'chinese', 'beijing', 'delhi',
    'modi', 'xi', 'jinping', 'border', 'government'
]
for w in custom_stopwords:
    nlp.vocab[w].is_stop = True
print("模型加载与配置完成。")

# 应用基础清洗
print("\n正在应用基础清洗...")
df['cleaned_text'] = df['content'].apply(basic_clean)

# 应用深度清洗 (Tokenization, Lemmatization, Stopword Removal)
print("正在应用深度清洗...")
texts_to_process = df['cleaned_text'].tolist()
token_list = []

# 在Linux环境下，使用 n_process=-1 可以利用所有CPU核心进行并行处理，以获得最高效率。
with tqdm(total=len(texts_to_process), desc="Spacy Processing") as pbar:
    for doc in nlp.pipe(texts_to_process, batch_size=50, n_process=-1):
        # 从处理好的 doc 对象中提取符合条件的词元
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and len(token.lemma_) > 2
        ]
        token_list.append(tokens)
        pbar.update(1)
# 将结果存回 DataFrame
df['tokens_for_lda'] = token_list

# N-grams构建
print("\n正在构建Bigrams和Trigrams...")
# 注意：这里的 token_list 已经是我们刚刚从 nlp.pipe 生成的结果
bigram = Phrases(token_list, min_count=5, threshold=100)
bigram_phraser = Phraser(bigram)
trigram = Phrases(bigram[token_list], threshold=100)
trigram_phraser = Phraser(trigram)
# 应用N-grams模型到我们的词元列表
df['tokens_for_lda'] = [trigram_phraser[bigram_phraser[doc]] for doc in tqdm(token_list, desc="Applying N-grams")]
print("\n预处理和N-grams构建完成。")
print(df[['cleaned_text', 'tokens_for_lda']].head())


# --- 6. 保存最终结果 ---
output_file = PROCESSED_DATA_DIR / "main_corpus.pkl"
df.to_pickle(output_file)
print(f"\n预处理完成的数据已成功保存到: {output_file}")
