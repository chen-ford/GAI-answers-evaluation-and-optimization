import pandas as pd
import pickle
import tensorflow as tf
import keras
from keras.models import load_model
import keras.src.models


# 注意力机制层
@keras.saving.register_keras_serializable(package="custom_objects")
class Attention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__()
        self.W = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, inputs):
        query = inputs[0]
        values = inputs[1]

        query_with_time_axis = tf.expand_dims(query, 1)

        score = tf.nn.tanh(self.W(query_with_time_axis) + self.W(values))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


# %%
# 读取xgboost模型
def load_model_pickle(model_name):
    with open(model_name, 'rb') as f:
        model_xg = pickle.load(f)
    return model_xg


xgboost = load_model_pickle(r'C:\Users\mi\PycharmProjects\textAnalysis\分类模型\ML_models_smote\xgboost_model_6.pkl')
# 读取bert_bi_lstm_attention模型
bert_bi_lstm_attention_model = load_model(r'C:\Users\mi\PycharmProjects\textAnalysis\分类模型\DL_models_smote\bert_bi'
                                          r'-lstm_attention_smote\bert_bilstm_attention_model_l.keras',
                                          custom_objects={'Attention': Attention})


# 组合模型
def compositional_model_predict(X_test_features, X_test_embedding, model1=xgboost, model2=bert_bi_lstm_attention_model,
                                r1=0.5, r2=0.5):
    # Predict probabilities
    y_pred_proba = pd.DataFrame(model1.predict_proba(X_test_features)[:, 1]) * r1 + pd.DataFrame(
        model2.predict(X_test_embedding) * r2)

    return y_pred_proba
