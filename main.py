from body import *
from parse import *
from train import *
from kNN import *
from neuralnet import *
from svm import *
import matplotlib.pyplot as plt
from decisionTree import *

def main():

  decision_tree("star_classification.csv")

  return
  file, alg, goal = parse_command_line()

  if alg == 'kNN':
    if goal == 'train':
      kNN_train_test('star_classification.csv')
    else:
      k = int(input("Enter the value of k: "))
      new_data = parse_new_data('new_data.csv')
      kNN_predict(k, new_data)
  elif alg == 'neuralNet':
    if goal == 'train':
      print('here')
      model, scaler = neural_network_train('star_classification.csv')
    else:
      model, scaler = neural_network_train('star_classification.csv')
      new_data = parse_new_data('new_data.csv')
      neural_network_predict(model, scaler, new_data)
  else:
    if goal == 'train':
      #SVM train
      model, scaler = SVM_train('star_classification.csv')
    else:
      #SVM predict
      model, scaler = SVM_train('star_classification.csv')
      new_data = parse_new_data('new_data.csv')
      SVM_predict(model, new_data, scaler)


  #SVM_train('star_classification.csv')

  #model, scaler = neural_network_train('star_classification.csv')

  # k, X, y, = kNN_train_test('star_classification.csv')

  #new_data = parse_new_data('new_data.csv')

  # kNN_predict(k, X, y, new_data)

  #model, scaler = SVM('star_classification.csv')

  #SVM_predict(model, new_data, scaler)



  #neural_network_predict(model, scaler, new_data)


  



if __name__ == '__main__':
  main()