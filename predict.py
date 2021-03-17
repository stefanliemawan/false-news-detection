import keras
import tensorflow
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def fakeTrue():
    # split into functions later
    predict_data = pd.read_csv('./cleanDatasets/clean_fake_true.csv')
    predict_data = shuffle(predict_data)
    model = keras.models.load_model('./models/256-10-model1.h5')
    titles = predict_data['title']
    # texts = predict_data['text'].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(predict_data["title"])
    sequences = tokenizer.texts_to_sequences(predict_data["title"])

    x1 = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=25)

    x2 = np.array([[0, 0] for x in titles])
    print(x1.shape)
    print(x2.shape)
    y1, y2 = model.predict([x1, x2])
    y1_classes = y1.argmax(axis=-1)
    y2_classes = y2.argmax(axis=-1)
    # print(y1_classes)
    # print(y2_classes)

    with open('label_mapping.json', 'r') as f:
        label_dict = json.load(f)
    with open('subj_mapping.json', 'r') as f:
        subj_dict = json.load(f)

    print(label_dict)
    print(subj_dict)

    ft_prediction = []

    for i in range(x1.shape[0]):
        text = predict_data['title'].iloc[i]
        ft_label = predict_data['label'].iloc[i].upper()
        label = label_dict[str(y1_classes[i])]
        subjectivity = subj_dict[str(y2_classes[i])]
        ft_prediction.append([text, ft_label, label, subjectivity])

    ft_prediction = pd.DataFrame(ft_prediction, columns=[
                                 "text", "ft_label", "label", "subjectivity"])
    print(ft_prediction)
    ft_prediction.to_csv('./prediction/faketrue_prediction.csv', index=False)


def main():


main()
