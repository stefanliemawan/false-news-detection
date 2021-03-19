import keras
import tensorflow
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def processInputTextOnly(texts, model):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    x1 = pad_sequences(
        sequences, padding='post', truncating='post', maxlen=25)

    x2 = np.array([[0, 0] for x in texts])
    print(x1.shape)
    print(x2.shape)
    y1, y2 = model.predict([x1, x2])
    y1_classes = y1.argmax(axis=-1)
    y2_classes = y2.argmax(axis=-1)
    # print(y1_classes)
    # print(y2_classes)

    return y1_classes, y2_classes


def getAccuracy(data):
    predicted_true = 0
    predicted_false = 0

    for index, row in data.iterrows():
        if row['label'] == row['predicted_label']:
            predicted_true += 1
        else:
            predicted_false += 1

    print("Label Predicted True, ", predicted_true)
    print("Label Predicted False ", predicted_false)

    acc = predicted_true / (predicted_true + predicted_false)
    print("Accuracy = ", acc)


def predictBinary(data, col_name):
    model = keras.models.load_model('./models/256-10-model1.h5')
    predict_data = []

    y1_classes, y2_classes = processInputTextOnly(data[col_name], model)

    with open('label_mapping.json', 'r') as f:
        label_dict = json.load(f)
    with open('subj_mapping.json', 'r') as f:
        subj_dict = json.load(f)

    print(label_dict)
    print(subj_dict)

    for i in range(data.shape[0]):
        text = data[col_name].iloc[i]
        label = data['label'].iloc[i].upper()
        predicted_label = label_dict[str(y1_classes[i])]
        subjectivity = subj_dict[str(y2_classes[i])]
        predict_data.append([text, label, predicted_label, subjectivity])

    predict_data = pd.DataFrame(predict_data, columns=[
        "text", "label", "predicted_label", "subjectivity"])

    predict_data.loc[predict_data.predicted_label ==
                     "MOSTLY-TRUE", "predicted_label"] = "TRUE"
    predict_data.loc[predict_data.predicted_label ==
                     "HALF-TRUE", "predicted_label"] = "TRUE"
    predict_data.loc[predict_data.predicted_label ==
                     "BARELY-TRUE", "predicted_label"] = "FALSE"
    predict_data.loc[predict_data.predicted_label ==
                     "PANTS-FIRE", "predicted_label"] = "FALSE"

    predict_data.loc[predict_data.label == "FAKE", "label"] = "FALSE"

    getAccuracy(predict_data)

    return predict_data


def fakeTrue():
    fake_true = pd.read_csv('./cleanDatasets/clean_fake_true.csv')
    fake_true = shuffle(fake_true)
    titles = fake_true['title']
    # texts = fake_true['text'].values

    predict_data = predictBinary(fake_true, 'title')
    predict_data.to_csv('./prediction/faketrue_prediction.csv', index=False)


def fakeNewsNet():
    politi_data = pd.read_csv('./cleanDatasets/fnn_politifact.csv')
    gossip_data = pd.read_csv('./cleanDatasets/fnn_gossip.csv')
    fnn_data = pd.concat([politi_data, gossip_data], ignore_index=True)

    fnn_data.loc[fnn_data.label == True, "label"] = "TRUE"
    fnn_data.loc[fnn_data.label == False, "label"] = "FALSE"
    fnn_data = shuffle(fnn_data)
    fnn_data['title'] = fnn_data['title'].astype(str)
    print(fnn_data.shape)

    predict_data = predictBinary(fnn_data, 'title')
    fnn_data.to_csv('./prediction/fnn_prediction.csv', index=False)


def main():
    # fakeTrue()
    fakeNewsNet()


main()
