from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertModel

# 检查GPU是否可用
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")  # 如果有多个GPU，使用GPU 1；否则使用第一个GPU
    # print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    # print("Using device: CPU")
# %%
# 加载BERT模型和tokenizer
config = BertConfig.from_json_file("../Bert_model/Bert/config.json")
tokenizer = BertTokenizer.from_pretrained('../Bert_model/Bert/')
bert = BertModel.from_pretrained("../Bert_model/Bert/", config=config)

bert.to(device)


# %%
def get_word_embeddings(text, model, token):
    text = "CLS" + str(text) + "SEP"
    encoded_inputs = token(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

    # Pretrain the text
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Get the CLS token embeddings
    # 将cls标记的向量作为整个句子向量
    word_embedding = outputs[0][:, 0, :]

    return word_embedding


# %%
# 获取词向量
def word_embeddings(data):
    x1 = get_word_embeddings(data[0], bert, tokenizer)
    data_embedding = x1
    for i in range(1, len(data)):
        x2 = get_word_embeddings(data[i], bert, tokenizer)
        data_embedding = torch.cat((x1, x2), dim=0)
        x1 = data_embedding

    return data_embedding

# %%
# 保存词向量
# torch.save(X_train_embedding, 'train_embedding.pt')
