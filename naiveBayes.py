import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def naive_bayes(file):

    df = pd.read_csv('star_classification.csv')

    X = df.drop(columns=['class'])
    y = df['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Build a Gaussian Classifier
    model = GaussianNB()

    # Model training
    model.fit(X_train, y_train)

    # Predict Output
    y_pred = model.predict(X_test)
    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="macro")

    print("Accuracy:", accuray)
    print("precision:", precision_score(y_pred, y_test, average='macro'))
    print('recall:', recall_score(y_pred, y_test, average='macro'))
    print("F1 Score:", f1)


    return


if __name__ == '__main__':
    naive_bayes("star_classification.csv")

