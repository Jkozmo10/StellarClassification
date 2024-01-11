# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def decision_tree(file):

    df_stars = pd.read_csv(file)
    labelEncoder = LabelEncoder()
    df_stars['class'] = labelEncoder.fit_transform(df_stars['class']) #Galaxy = 0, qso = 1, star = 2
    
    X = df_stars.drop(columns=['class'])
    y = df_stars['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # model = DecisionTreeClassifier(max_depth=9)
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # print(precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred,  average='macro'), f1_score(y_test, y_pred, average='macro'))

    gs_model = GridSearchCV(estimator=DecisionTreeClassifier(), 
                            param_grid={'max_depth' : np.arange(2, 20, 1)}, 
                            verbose=3,
                            scoring='f1_macro',
                            cv=10)

    gs_model.fit(X_train, y_train)

    results = pd.DataFrame(gs_model.cv_results_)

    # print(results.plot.scatter(x="param_max_depth", y="mean_test_score"))

    plt.plot(results["param_max_depth"], results["mean_test_score"], 'bo')
    plt.xlabel("Max Depth")
    plt.ylabel('F1-Score')
    plt.xticks(np.arange(2, 20, 1), np.arange(2, 20, 1))
    plt.show()


    # plt.plot(gs_model.cv_results_['mean_test_score'], label='Mean test score')
    # plt.xlabel('Parameter')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.show()


    print(gs_model.best_params_)





    return




    

