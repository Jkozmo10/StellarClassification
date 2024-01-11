import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def neural_network_train(file):
    df = pd.read_csv(file)
    labelEncoder = LabelEncoder()
    df['class'] = labelEncoder.fit_transform(df['class']) #Galaxy = 0, qso = 1, star = 2
    
    X = df.drop(columns=['class'])
    y = df['class'].values

    scaler = StandardScaler()

    X = scaler.fit_transform(X.values)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    y_train = tf.keras.utils.to_categorical(y_train, 3)
    print(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test, 3)

    tf.random.set_seed(66)

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(15, activation='relu'),
    # tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    )

    history = model.fit(X_train, y_train, epochs=100, batch_size=500)

    # score = model.evaluate(X_test, y_test, batch_size=500)
    print(np.argmax(model.predict(X_test), axis=1))
    print(y_test)

    print(precision_score(y_test, np.argmax(model.predict(X_test), axis=1), average='macro'), recall_score(y_test, np.argmax(model.predict(X_test), axis=1),  average='macro'), f1_score(y_test, np.argmax(model.predict(X_test), axis=1), average='macro'))

    return model, scaler

def neural_network_predict(model, scaler, new_data):
    new_data = scaler.transform(new_data)

    output = []

    predictions = model.predict(
        x=new_data,
        batch_size= 1,
        verbose=0
    )

    rounded = np.argmax(predictions, axis=1)

    for pred in rounded:
        if pred == 0:
            output.append('GALAXY')
        elif pred == 1:
            output.append('QSO')
        else:
            output.append('STAR')

    print(output)


if __name__ == '__main__':
    neural_network_train("star_classification.csv")





