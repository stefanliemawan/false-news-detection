import pandas as pd


def main():
    liar_train = pd.read_csv(
        './datasets/LIAR/liar_train_labeled.csv').reset_index(drop=True)
    liar_test = pd.read_csv(
        './datasets/LIAR/liar_test_labeled.csv').reset_index(drop=True)
    liar_val = pd.read_csv(
        './datasets/LIAR/liar_valid_labeled.csv').reset_index(drop=True)
    politi = pd.read_csv(
        './datasets/PolitiFact/politifact.csv').reset_index(drop=True)
    data = pd.concat([liar_train, liar_test, liar_val,
                      politi])
    data['statement'] = data['statement'].replace({'"': ''}, regex=True)
    data = data.reset_index(drop=True)
    data = data.drop(['id', 'subject', "speaker's job title",
                      'state', 'party', 'context', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', 'date', 'checker'], axis=1)

    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = data.drop_duplicates(subset="statement")
    print(data)
    print(data.shape)
    data.to_csv('./datasets/merged_politifact.csv', index=False)


main()
