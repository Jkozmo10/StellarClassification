from body import *
from parse import *
from train import *
from kNN import *
from neuralnet import *
import torch

def main():
  # bodies = parse_file("star_classification.csv")
  # trainingData = createTrainingData(bodies)
  # testingData = createTestingData(bodies, trainingData)

  # for body in trainingData:
  #   print(body)

  # print("testing data: ")

  # for body in testingData:
  #   print(body)
  kNN('star_classification.csv')

  #neural_network('star_classification.csv')
  



if __name__ == '__main__':
  main()