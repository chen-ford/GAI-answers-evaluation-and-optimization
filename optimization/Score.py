# 自定义文件
from AnswerFeatures import feature
from WordVector import word_embeddings
from CompositionalModel import compositional_model_predict

from sklearn.preprocessing import MinMaxScaler
import torch
from rouge import Rouge
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import jieba

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


def score(question, answer):
    answer_feature = feature(question, answer)
    scaler = MinMaxScaler()
    answer_feature_scaled = scaler.fit_transform(answer_feature)

    answer_embedding = word_embeddings(answer)
    answer_embedding = torch.unsqueeze(torch.tensor(answer_embedding), dim=1).numpy()

    predict_prob = compositional_model_predict(answer_feature_scaled, answer_embedding)

    return predict_prob


# 计算rouge-1值的函数
def rouge_1(summaries, references):
    rouge = Rouge()
    all_rouge_scores = []  # 储存所有摘要对的ROUGE得分

    for summary, reference in zip(summaries, references):
        pred_reference = ' '.join(jieba.lcut(reference))
        pred_summary = ' '.join(jieba.lcut(summary))
        # 去除停用词
        pred_reference = ' '.join([word for word in pred_reference if word not in stopwords])
        pred_summary = ' '.join([word for word in pred_summary if word not in stopwords])
        scores = rouge.get_scores(pred_summary, pred_reference)
        # print(scores)
        all_rouge_scores.append(scores[0]['rouge-1']['f'])

    avg_rouge_1_f_score = sum(all_rouge_scores) / len(all_rouge_scores)

    return avg_rouge_1_f_score


def rouge_2(summaries, references):
    rouge = Rouge()
    all_rouge_scores = []  # 储存所有摘要对的ROUGE得分

    for summary, reference in zip(summaries, references):
        pred_reference = ' '.join(jieba.lcut(reference))
        pred_summary = ' '.join(jieba.lcut(summary))
        # 去除停用词
        pred_reference = ' '.join([word for word in pred_reference if word not in stopwords])
        pred_summary = ' '.join([word for word in pred_summary if word not in stopwords])
        scores = rouge.get_scores(pred_summary, pred_reference)
        # print(scores)
        all_rouge_scores.append(scores[0]['rouge-2']['f'])

    avg_rouge_2_f_score = sum(all_rouge_scores) / len(all_rouge_scores)

    return avg_rouge_2_f_score


def rouge_l(summaries, references):
    rouge = Rouge()
    all_rouge_scores = []  # 储存所有摘要对的ROUGE得分

    for summary, reference in zip(summaries, references):
        pred_reference = ' '.join(jieba.lcut(reference))
        pred_summary = ' '.join(jieba.lcut(summary))
        # 去除停用词
        pred_reference = ' '.join([word for word in pred_reference if word not in stopwords])
        pred_summary = ' '.join([word for word in pred_summary if word not in stopwords])
        scores = rouge.get_scores(pred_summary, pred_reference)
        # print(scores)
        all_rouge_scores.append(scores[0]['rouge-l']['f'])

    avg_rouge_l_f_score = sum(all_rouge_scores) / len(all_rouge_scores)

    return avg_rouge_l_f_score


def bleu_4(gen_texts, ref_texts):
    bleu_scores = []
    smoother = SmoothingFunction().method3
    for gen_text, ref_text in zip(gen_texts, ref_texts):
        # ref_texts应为一个包含多个参考文本的列表
        pred_reference = jieba.lcut(ref_text)
        pred_summary = jieba.lcut(gen_text)
        # 去除停用词
        pred_reference = [[word for word in pred_reference if word not in stopwords]]
        # print("pred_reference:", pred_reference)
        pred_summary = [word for word in pred_summary if word not in stopwords]
        # print("pred_summary:", pred_summary)
        scores = sentence_bleu(pred_reference, pred_summary, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)  # BLEU-4
        bleu_scores.append(scores)
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score
