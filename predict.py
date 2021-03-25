import keras
import tensorflow
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocess import cleanDataText


def processInputTextOnly(texts, model):
    tokenizer = Tokenizer(num_words=19129)
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


def getAccuracy(predict_data):
    predicted_true = 0
    predicted_false = 0

    for index, row in predict_data.iterrows():
        if row['label'] == row['predicted_label']:
            predicted_true += 1
        else:
            predicted_false += 1

    print('Label Predicted True, ', predicted_true)
    print('Label Predicted False ', predicted_false)

    acc = predicted_true / (predicted_true + predicted_false)
    print('Accuracy = ', acc)


def predict(data, col_name, label, binary):
    # model = keras.models.load_model('./models/256-10-model1.h5')
    model = keras.models.load_model('./models/128-20-no-duplicate-model1.h5')
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
        predicted_label = label_dict[str(y1_classes[i])]
        subjectivity = subj_dict[str(y2_classes[i])]
        if label == True:
            label = data['label'].iloc[i].upper()
            predict_data.append([text, label, predicted_label, subjectivity])
        else:
            predict_data.append([text, predicted_label, subjectivity])

    if label == True:
        predict_data = pd.DataFrame(predict_data, columns=[
            'text', 'label', 'predicted_label', 'subjectivity'])
    else:
        predict_data = pd.DataFrame(predict_data, columns=[
            'text', 'predicted_label', 'subjectivity'])

    if binary == True:
        predict_data.loc[predict_data.predicted_label ==
                         'MOSTLY-TRUE', 'predicted_label'] = 'TRUE'
        predict_data.loc[predict_data.predicted_label ==
                         'HALF-TRUE', 'predicted_label'] = 'TRUE'
        predict_data.loc[predict_data.predicted_label ==
                         'BARELY-TRUE', 'predicted_label'] = 'FALSE'
        predict_data.loc[predict_data.predicted_label ==
                         'PANTS-FIRE', 'predicted_label'] = 'FALSE'

        predict_data.loc[predict_data.label == 'FAKE', 'label'] = 'FALSE'

    print(predict_data['predicted_label'].value_counts())

    if label == True:
        getAccuracy(predict_data)

    return predict_data


def fakeTrue():
    fake_true = pd.read_csv('./cleanDatasets/clean_fake_true.csv')
    fake_true = shuffle(fake_true)
    titles = fake_true['title']
    # texts = fake_true['text'].values

    predict_data = predict(fake_true, 'title', label=True, binary=True)
    predict_data.to_csv('./prediction/faketrue_prediction.csv', index=False)

    # 0.47 Prediction Accuracy


def fakeNewsNet():
    politi_data = pd.read_csv('./cleanDatasets/fnn_politifact.csv')
    gossip_data = pd.read_csv('./cleanDatasets/fnn_gossip.csv')
    fnn_data = pd.concat([politi_data, gossip_data], ignore_index=True)

    fnn_data.loc[fnn_data.label == True, 'label'] = 'TRUE'
    fnn_data.loc[fnn_data.label == False, 'label'] = 'FALSE'
    fnn_data = shuffle(fnn_data)
    fnn_data['title'] = fnn_data['title'].astype(str)
    print(fnn_data.shape)

    predict_data = predict(fnn_data, 'title', label=True, binary=True)
    fnn_data.to_csv('./prediction/fnn_prediction.csv', index=False)

    # 0.47 Prediction Accuracy


def nytimes():
    ny_data = pd.DataFrame(['Louisiana House Race Sets Up a Democratic Showdown in New Orleans',
                            'Sexual Anguish of Atlanta Suspect Is Familiar Thorn for Evangelicals', 'Five Who Used Marijuana in Past Will Exit White House, Calling New Guidelines Into Question', 'Assaulting the Truth, Ron Johnson Helps Erode Confidence in Government', 'Louisiana House Race Sets Up a Democratic Showdown in New Orleans', 'Defense Secretary Austin Make Unannounced Visit to Afghanistan', 'Louisiana House Race Sets Up a Democratic Showdown in New Orleans', 'Biden and Harris Condemn Violence Against Asian-Americans', 'Confronting Violence Against Asians, Biden Says That We Cannot Be Complicit', 'Reed Disputes Groping Allegation, Calling Woman’s Account Not Accurate', 'Stay Scattered and Avoid Police, Proud Boys Were Told Before Capitol Riot'], columns=['statement'])
    ny_data = cleanDataText(ny_data, 'statement')
    predict_data = predict(ny_data, 'statement', label=False, binary=False)
    print(predict_data)


def newsCategoryDataset():
    ncd_data = pd.read_json(
        './datasets/News_Category_Dataset_v2/News_Category_Dataset_v2.json', lines=True)
    ncd_data = cleanDataText(ncd_data, 'headline')
    print(ncd_data)
    predict_data = predict(ncd_data, 'headline', label=False, binary=False)


def main():
    # fakeTrue()
    # fakeNewsNet()
    nytimes()
    # newsCateg oryDataset()


main()
