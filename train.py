from body import *
import random

def createTrainingData(data):
    trainingData = {}
    numbers = random.sample(range(len(data)), int(len(data) * 0.8))

    for rInt in numbers:
        trainingData[rInt] = data[rInt]
    
    return trainingData

def createTestingData(data, trainingData):
    testingData = {}

    for key in data:
        if key not in trainingData:
            testingData[key] = data[key]

    return testingData

def createCrossValidationSets():
    pass