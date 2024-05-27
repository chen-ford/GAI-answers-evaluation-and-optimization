# 特征处理准备
# %%
import pandas as pd
import math
import os
import re
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig

# 加载本地Bert模型

# Load the BERT model and tokenizer
config = BertConfig.from_json_file("../Bert_model/Bert/config.json")
tokenizer = BertTokenizer.from_pretrained('../Bert_model/Bert/')
bert = BertModel.from_pretrained("../Bert_model/Bert/", config=config)


# 获取词向量
def get_word_embeddings(text, model, token=tokenizer):
    text = "CLS" + text + "SEP"
    encoded_inputs = token(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

    # Pretrain the text
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Get the CLS token embeddings
    # 将cls标记的向量作为整个句子向量
    word_embeddings = outputs[0][:, 0, :]

    return word_embeddings


# %%
# 读取停用词
folder_path = r"C:\Users\mi\PycharmProjects\textAnalysis\特征指标量化\stopwords"
stopwords = []
try:
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                s = file.readlines()
                # print(f"文件名: {file_name}")
                stopwords += s

                # print("----------------------")
except FileNotFoundError:
    print("找不到指定的文件夹，请检查文件夹路径是否正确。")
except Exception as e:
    print("发生错误:", e)
stopwords = [word.replace('\n', '') for word in stopwords]
# print(stopwords)

# %%
# 医学术语列表
def Medicalwordslist(filepath2):  # 创建专业医学词汇列表
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Medicalword = [re.sub(pattern, "", line.strip().replace(" ", '')) for line in
                   open(filepath2, 'r', encoding="utf-8").readlines()]  # 以行的形式读取停用词表，同时转换为列表
    return Medicalword


medicalword = Medicalwordslist(r'C:\Users\mi\PycharmProjects\textAnalysis\medical-term\all_medical_terms.txt')


# 特征提取函数
# %%
# 文本长度
def count_text_length(text):
    return len(text)


# %%
# 句子数
def count_num_sentences(text):
    # 使用正则表达式匹配中文句号、问号、感叹号作为句子的结束符号
    pattern = r'[\u4e00-\u9fa5][。？！]'
    sentences = re.findall(pattern, text)

    # 最后一句可能没有结束符号，需要额外判断
    if len(text) > 0 or text[-1] in '。？！':
        sentences.append(text[-1])

    return len(sentences)


# %%
# 单词数量
def count_num_words(text):
    # 使用正则表达式去除标点符号
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)

    # 去除空格等无意义字符
    words = [word for word in words if word.strip()]

    return len(words)


# %%
# 停用词数
def count_num_stopwords(text, stopword=None):
    # 使用正则表达式去除标点符号
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)
    # 去除空格等无意义字符
    words = [word for word in words if word.strip()]
    # 计算停用词数
    stop_words = [word for word in words if word in stopword]
    return len(stop_words)


# %%
# 标点符号数
def count_num_punctuation(text):
    # 常见的中文标点符号列表
    chinese_punctuation = '，。！？；：“”、（）《》「」【】『』｛｝……'

    # 初始化计数器
    count = 0

    # 遍历文本的每个字符
    for char in text:
        # 如果字符在中文标点符号列表中，则计数器加1
        if char in chinese_punctuation:
            count += 1

    # 返回标点符号数量
    return count


# %%
# 短句数量
def count_num_short_sentences(text, max_length=20):
    # 句末标点符号列表
    sentence_ending_punctuation = '，。！？；：“”、（）《》「」【】『』｛｝……'

    # 使用正则表达式来匹配句末标点符号，并分割文本为句子列表
    sentences = re.split(r'[{}]+'.format(re.escape(sentence_ending_punctuation)), text)
    sentences = sentences[:-1]
    # 过滤并计数字符数在max_length以下的句子
    short_sentence_count = sum([1 for x in sentences if len(x.strip()) <= max_length])

    # 返回短句数量
    return short_sentence_count


# %%
# 形容词数
def count_num_adjectives(text):
    # 使用jieba的词性标注功能
    words = pseg.cut(text)

    # 定义形容词的词性标签集合
    # 'a' 代表形容词
    adjective_tags = ['a', 'ad', 'an', 'ag', 'al']

    # 初始化形容词计数器
    adjective_count = 0

    # 遍历每个词和对应的词性
    for word, flag in words:
        # 如果词性标签在形容词标签集合中，则增加计数器
        if flag in adjective_tags:
            adjective_count += 1

            # 返回形容词数量
    return adjective_count


# %%
# 副词数
def count_num_adverbs(text):
    # 使用jieba的词性标注功能
    words = pseg.cut(text)

    # 定义副词的词性标签
    adverb_tags = ['d', 'df', 'dg']

    # 初始化副词计数器
    adverb_count = 0

    # 遍历每个词和对应的词性
    for word, flag in words:
        # 如果词性标签是副词标签，则增加计数器
        if flag in adverb_tags:
            adverb_count += 1

            # 返回副词数量
    return adverb_count


# %%
# 名词数
def count_num_nouns(text):
    # 使用jieba进行分词和词性标注
    words = pseg.cut(text)

    noun_count = 0
    noun = ['n', 'nr', 'nr1', 'nr2', 'nrj', 'ns', 'nsf', 'nt', 'nz', 'nl', 'nx', 'ng', 'nrt', 'nrfg']
    # 遍历分词结果，统计名词数量
    for word, flag in words:
        # 'n' 在jieba中表示名词
        if flag in noun:
            noun_count += 1

    return noun_count


# %%
# 动词数
def count_num_verbs(text):
    # 使用jieba的词性标注功能
    words = pseg.cut(text)

    # 定义动词的词性标签
    verb = ['v', 'vd', 'vg', 'vi', 'vn', 'vq', 'vshi', 'vyou', 'vf', 'vx', 'vl']

    # 初始化动词计数器
    verb_count = 0

    # 遍历每个词和对应的词性
    for word, flag in words:
        # 如果词性标签是动词标签，则增加计数器
        if flag in verb:
            verb_count += 1

            # 返回动词数量
    return verb_count


# %%
# 答案关键词数
def count_num_keywords(text, stopword=None):
    # 使用正则表达式去除标点符号
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)

    # 去除停用词
    words = [word for word in words if word.strip() and word not in stopword]

    return len(words)


# %%
# 文本信息熵
def calculate_entropy(text, stopword=None):
    # 使用正则表达式去除标点符号
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)
    # 去除停用词
    words = list(set([word for word in words if word.strip() and word not in stopword]))

    char_freq = {}
    total_chars = 0
    entropy = 0.0

    # 统计每个字符出现的频率
    for char in words:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
        total_chars += 1

    # 计算信息熵
    for freq in char_freq.values():
        probability = freq / total_chars
        entropy -= probability * math.log(probability, 2)

    return round(entropy, 4)


# %%
# 医学术语数
def count_num_medicalwords(text, stopword=None, medical_word=None):
    if medical_word is None:
        medical_word = medical_word
    if stopword is None:
        stopword = stopwords
    Medical_word = medical_word
    # 去除标点符号和停用词
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)

    # 去除停用词
    words = list(set([word for word in words if word.strip() and word not in stopword]))

    medical_words = []  # 医学专业词的总长度
    for word in words:
        if word in Medical_word:
            # print("word:", word)
            medical_words.append(word)

    return len(medical_words)


# %%
# 医学术语与文本关键词数比值
def density_medicalwords(num_keywords, num_medicalwords):
    return round(num_medicalwords / num_keywords, 3) if num_keywords > 0 else 0


# %%
# 问题和答案关键词重合度
def extract_keywords(text, stopword=None):
    # 使用正则表达式去除标点符号
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text, cut_all=True)

    # 去除停用词
    words = [word for word in words if word.strip() and word not in stopword]
    # print(words)
    return words


def calculate_keyword_overlap(text1, text2):
    """
    计算两段文本的关键词重合度
    :param text1: 第一段文本
    :param text2: 第二段文本
    :return: 关键词重合度
    """
    # 提取关键词
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    # 计算重合的关键词数量
    overlap = len(set(keywords1) & set(keywords2))

    # 计算总的关键词数量（去除重复）
    total = len(set(keywords1) | set(keywords2))

    # 计算重合度
    overlap_rate = overlap / total if total > 0 else 0

    return round(overlap_rate, 3)


# %%
# 问题和答案语义相似性
def calculate_semantic_similarity(question, answer, model=bert, token=tokenizer):
    a = 0
    try:
        s1 = get_word_embeddings(question, model, token)
        s2 = get_word_embeddings(answer, model, token)
        # 计算余弦相似度
        similarity = cosine_similarity(s1.cpu(), s2.cpu())
        a = round(np.mean(np.mean(similarity, axis=1)), 4)
        # print(f"第{i+1}段文本的语义相似度为: {a:.4f}")
    except Exception as e1:
        print(f"相似度计算的错误为{e1}")

    return a


# %%
# 积极情感词数
def count_num_positive_words(text, stopword=None):
    # 去除标点符号和停用词
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [word for word in words if word.strip() and word not in stopword]

    # 初始化积极情感词计数器
    positive_word_count = 0

    # 遍历每个词，尝试进行情感分析
    for word in words:
        # 注意：这里假设SnowNLP的sentiment方法返回的情感分数越高，表示越积极
        # 但实际上，SnowNLP的sentiment方法通常用于判断整个句子的情感倾向，而不是单个词
        sentiment_score = SnowNLP(word).sentiments

        # 设定一个阈值，认为情感分数大于这个阈值的词为积极情感词
        threshold = 0.5  # 这个阈值需要根据实际情况进行调整
        if sentiment_score > threshold:
            # print(word)
            positive_word_count += 1

    return positive_word_count


# %%
# 消极情感词数
def count_num_negative_words(text, stopword=None):
    if stopword is None:
        stopword = stopwords
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 使用结巴分词对中文文本进行分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [word for word in words if word.strip() and word not in stopword]

    # 初始化积极情感词计数器
    negative_word_count = 0

    # 遍历每个词，尝试进行情感分析
    for word in words:
        # 注意：这里假设SnowNLP的sentiment方法返回的情感分数越高，表示越积极
        sentiment_score = SnowNLP(word).sentiments

        # 设定一个阈值，认为情感分数大于这个阈值的词为积极情感词
        threshold = 0.5  # 这个阈值需要根据实际情况进行调整
        if sentiment_score < threshold:
            # print(word)
            negative_word_count += 1

    return negative_word_count


# %%
# 情感极性
def calculate_sentiment_polarity(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    s1 = SnowNLP(text)

    return round(s1.sentiments, 5)


# 特征量化
# %%
# 函数进行特征提取
def feature_extraction(question, answer):
    # 文本长度
    len_text = count_text_length(answer)
    # 句子数
    num_sentences = count_num_sentences(answer)
    # 单词数
    num_words = count_num_words(answer)
    # 停用词数量
    num_stopwords = count_num_stopwords(answer)
    # 标点符号数
    num_punctuations = count_num_punctuation(answer)
    # 短句数
    num_short_sentences = count_num_short_sentences(answer)
    # 形容词数
    num_adjectives = count_num_adjectives(answer)
    # 动词数
    num_verbs = count_num_verbs(answer)
    # 名词数
    num_nouns = count_num_nouns(answer)
    # 动词数
    num_adverbs = count_num_adverbs(answer)
    # 关键词数
    num_keywords = count_num_keywords(answer)
    # 文本信息熵
    text_entropy = calculate_entropy(answer)
    # 医学术语数量
    num_medical_terms = count_num_medicalwords(answer)
    # 医学术语与关键词比例
    medical_keywords_ratio = density_medicalwords(num_keywords, num_medical_terms)
    # 问题和答案关键词重合度
    keywords_overlap = calculate_keyword_overlap(question, answer)
    # 语义相似度
    semantic_similarity = calculate_semantic_similarity(question, answer)
    # 积极情感词数
    num_positive_words = count_num_positive_words(answer)
    # 消极情感词数
    num_native_words = count_num_negative_words(answer)
    # 情感极性
    sentiment_polarity = calculate_sentiment_polarity(answer)

    return [len_text, num_sentences, num_words, num_stopwords, num_punctuations, num_short_sentences, num_adjectives,
            num_verbs, num_nouns, num_adverbs, num_keywords, text_entropy, num_medical_terms, medical_keywords_ratio,
            keywords_overlap, semantic_similarity, num_positive_words, num_native_words, sentiment_polarity]


# %%
# 读取数据
def feature(data_question, data_answer):
    # 调用函数进行特征提取
    df = pd.DataFrame(
        columns=['len_text', 'num_sentences', 'num_words', 'num_stopwords', 'num_punctuations', 'num_short_sentences',
                 'num_adjectives', 'num_verbs', 'num_nouns', 'num_adverbs', 'num_keywords', 'text_entropy',
                 'num_medical_terms', 'medical_keywords_ratio', 'keywords_overlap', 'semantic_similarity',
                 'num_positive_words', 'num_native_words', 'sentiment_polarity'])
    for i in range(len(data_question)):
        try:
            df.loc[i] = feature_extraction(data_question[i], data_answer[i])
        except Exception as e2:
            df.loc[i] = 0
            print(e2)

    # df.to_csv('data_feature.csv')
    return df
