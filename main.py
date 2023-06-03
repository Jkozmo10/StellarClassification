from body import *
from parse import *
from train import *
from kNN import *
from neuralnet import *
from svm import *
import matplotlib.pyplot as plt

def main():
  # bodies = parse_file("star_classification.csv")
  # trainingData = createTrainingData(bodies)
  # testingData = createTestingData(bodies, trainingData)

  # for body in trainingData:
  #   print(body)

  # print("testing data: ")

  # for body in testingData:
  #   print(body)

  # plt.plot([1,2,3,4,5], [10,23,56,79,90], 'bo')
  # plt.xlabel("k")
  # plt.ylabel('Accuracy')
  # plt.show()
  kNN_train('star_classification.csv')

  #SVM('star_classification.csv')

  #neural_network('star_classification.csv')
  



if __name__ == '__main__':
  main()