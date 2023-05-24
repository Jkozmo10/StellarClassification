import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def kNN(file):
    df = pd.read_csv(file)
    features = df.drop(columns=['classification'])
    label = df['classification'].values

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=1, stratify=label)

    # kNN = KNeighborsClassifier(n_neighbors = 3)
    # kNN.fit(X_train,y_train)

    for k in range(1, 20):
        kNNCross = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(kNNCross, features, label, cv=10)
        #print(cv_scores)
        print('k:', k, 'cv_mean scores:', np.mean(cv_scores))


    #print(kNN.score(X_test, y_test))