# libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(filename):
    return pd.read_csv(filename)


def pre_processing(df, number_of_feature, is_train_model=False):
    df.drop(['X3', 'X31', 'X32', 'X127', 'X128', 'X590'], axis=1)

    # seperate label and features
    x = df.loc[:, df.columns[0:-1]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    y = None
    if is_train_model:
        y = df.loc[:, df.columns[-1]].values

    # PCA
    pca = PCA(n_components=number_of_feature)
    extracted_features = pca.fit_transform(x)
    extracted_df = pd.DataFrame(data=extracted_features)
    return extracted_df, y


def train_model(df, y, test_df):
    model = LogisticRegression()
    model.fit(df, y)
    return model.predict(test_df)


def write_output(predicted_list):
    with open('submission.csv', mode='w') as predicted_file:
        submission = csv.writer(predicted_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        submission.writerow(['ID', 'Predicted'])
        a = 1
        for i in predicted_list:
            submission.writerow([str(a), i])
            a = a + 1


def calc_accurancy(predicted_list):
    grand_truth_label = [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                         0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    accuracy = 0

    for i in range(80):
        if i not in [1, 2, 4, 7, 10, 13, 14, 21, 22, 24, 26, 28, 29, 32, 33, 38, 39, 40, 42, 44, 46, 47, 50, 51, 53,
                     54, 56, 59, 61, 63, 64, 67, 69, 71, 72, 73, 75, 77, 78, 79]:
            accuracy += grand_truth_label[i] == predicted_list[i]
    return accuracy / 40 * 100


train_df = load_data('train.csv')
test_df = load_data('test.csv')

extracted_df, y = pre_processing(train_df, 22, is_train_model=True)
test_df, y_none = pre_processing(test_df, 22)

predicted_list = train_model(extracted_df, y, test_df)
print(calc_accurancy(predicted_list))

