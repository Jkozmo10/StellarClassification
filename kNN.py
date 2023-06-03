import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def kNN_train(file):

    df = pd.read_csv('star_classification.csv')

    X = df.drop(columns=['class'])
    y = df['class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k_values = [i for i in range(1,25)]
    scores = []

    # new_data = [[1237680241455726592,12.2725745472949,-8.26055809287959,19.3772,17.42771,16.41689,16.00279,15.66193,8095,301,2,381,8055928843641575424,0.1137839,7155,56629,418]]
    # new_data = scaler.transform(new_data) #NEW PREDICTIONS

    for k in k_values:
        kNNCross = KNeighborsClassifier(n_neighbors=k)
        #kNNCross.fit(X, y) THIS IS HOW YOU GET NEW PREDICTIONS
        #print(k, kNNCross.predict(new_data))
        #print('before')
        cv_scores = cross_val_score(kNNCross, X, y, cv=10)
        #print('after')
        #print(cv_scores)
        scores.append(np.mean(cv_scores))
        print('k:', k, 'cv_mean scores:', np.mean(cv_scores))


    plt.plot(k_values, scores, 'bo')
    plt.xlabel("k")
    plt.ylabel('Accuracy')
    plt.show()