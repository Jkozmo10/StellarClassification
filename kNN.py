import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score



def kNN_train_test(file):

    df = pd.read_csv('star_classification.csv')

    # df = df[:5000] #for demo

    X = df.drop(columns=['class'])
    y = df['class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # k_values = [i for i in range(1,25)]
    # scores = []

    # for k in k_values:
    #     kNNCross = KNeighborsClassifier(n_neighbors=k)
    #     cv_scores = cross_val_score(kNNCross, X_train, y_train, cv=10)
    #     scores.append(np.mean(cv_scores))
    #     print('k:', k, 'cv_mean scores:', np.mean(cv_scores))


    # plt.plot(k_values, scores, 'bo')
    # plt.xlabel("k")
    # plt.ylabel('Accuracy')
    # plt.show()

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(precision_score(y_pred, y_test), recall_score(y_pred, y_test), f1_score(y_pred, y_test))

    # return scores.index(max(scores)) + 1


def kNN_predict(k, new_data):

    df = pd.read_csv('star_classification.csv')

    df = df[:5000] #for demo

    X = df.drop(columns=['class'])
    y = df['class']

    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)
    new_data = scaler.transform(new_data)

    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X, y)

    y_pred = kNN.predict(new_data)

    print(y_pred)

    return


if __name__ == '__main__':
    kNN_train_test("star_classification.csv")
