from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import metrics

def SVM_train(file):
    df = pd.read_csv(file)
    labelEncoder = LabelEncoder()
    df['class'] = labelEncoder.fit_transform(df['class']) #Galaxy = 0, qso = 1, star = 2

    df = df[:5000] #for demo only

    X = df.drop(columns=['class']) #features
    y = df['class'].values #label

    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    #same split, because there is uneven distribution of labels, we will stratify


    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
    linear = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)
    linear_pred = linear.predict(X_test)

    accuracy_lin = linear.score(X_test, y_test)
    accuracy_poly = poly.score(X_test, y_test)
    accuracy_rbf = rbf.score(X_test, y_test)
    print("Accuracy Linear Kernel:", accuracy_lin)
    print("Accuracy Polynomial Kernel:", accuracy_poly)
    print("Accuracy Radial Basis Kernel:", accuracy_rbf)

    return linear, scaler

def SVM_predict(model, new_data, scaler):

    new_data = scaler.transform(new_data)

    output = []

    predictions = model.predict(new_data)

    for pred in predictions:
        if pred == 0:
            output.append('GALAXY')
        elif pred == 1:
            output.append('QSO')
        else:
            output.append('STAR')


    print(output)


