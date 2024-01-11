import argparse

def parse_command_line():
  parser = argparse.ArgumentParser()

  parser.add_argument(dest = 'filename')
  parser.add_argument('-t', dest = 'goal', choices=['predict', 'train'], required=True)
  parser.add_argument('-a', dest = 'alg', choices=['kNN', 'neuralNet', 'SVM'], required=True)

  args = parser.parse_args()

  return args.filename, args.alg, args.goal

def parse_new_data(dataFile):
  inFile = open(dataFile, 'r')

  data = []

  for line in inFile:
    lst = line.strip().split(',')
    data.append(lst)
  
  return data

