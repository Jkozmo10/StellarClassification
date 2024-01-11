# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def random_forest(file):

    df_stars = pd.read_csv(file)
    labelEncoder = LabelEncoder()
    df_stars['class'] = labelEncoder.fit_transform(df_stars['class']) #Galaxy = 0, qso = 1, star = 2
    
    X = df_stars.drop(columns=['class'])
    y = df_stars['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

    # Create a random forest classifier
    rf = RandomForestClassifier(n_estimators=91, max_depth=10)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print(precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred,  average='macro'), f1_score(y_test, y_pred, average='macro'))




    return

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=10)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)


    # print(results.plot.scatter(x="param_max_depth", y="mean_test_score"))

    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

#    Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)

    # Best hyperparameters: {'max_depth': 10, 'n_estimators': 91}

    # plt.plot(results["param_max_depth"], results["mean_test_score"], 'bo')
    # plt.xlabel("Max Depth")
    # plt.ylabel('Score')
    # plt.xticks(np.arange(2, 20, 1), np.arange(2, 20, 1))
    # plt.show()


    # plt.plot(gs_model.cv_results_['mean_test_score'], label='Mean test score')
    # plt.xlabel('Parameter')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.show()


    # print(gs_model.best_params_)





    return

if __name__ == '__main__':
    random_forest("star_classification.csv")




    

