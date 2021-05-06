import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub
tf.gfile = tf.io.gfile


module_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

liar_train = pd.read_csv(
    "./cleanDatasets/clean_liar_train.csv").reset_index(drop=True)
liar_train = liar_train[liar_train["false counts"].notna()]
liar_train.fillna("None", inplace=True)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
    trainable=True)
outputs = encoder(encoder_inputs)

pooled_output = outputs["pooled_output"]      # [batch_size, 128].
sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 128].
embedding_model = tf.keras.Model(text_input, pooled_output)
sentences = tf.constant(["(your text here)"])
print(embedding_model(sentences))
