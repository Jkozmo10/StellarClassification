import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def kNN(file):
    df = pd.read_csv(file)
    X = df.drop(columns=['classification', 'obj_ID'])
    y = df['classification'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    scaler = StandardScaler()

    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # kNN = KNeighborsClassifier(n_neighbors = 3)
    # kNN.fit(X_train,y_train)

    k_values = [i for i in range(1, 4)]
    scores = []

    for k in range(1,4):
        print(k)
        kNNCross = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(kNNCross, X, y, cv=10)
        print('k', k, 'score:', np.mean(cv_scores))
        #scores.append(np.mean(cv_scores))
        print('end')

    print(scores)

    # plt.plot(k_values, scores, 'bo')
    # plt.xlabel("k")
    # plt.ylabel('Accuracy')
    # plt.show()


    #print(kNN.score(X_test, y_test))