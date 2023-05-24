import pandas as pd
import numpy as np

def neural_network(file):
    df = pd.read_csv(file)
    features = df.drop(columns=['classification'])
    label = df['classification'].values

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=1, stratify=label)