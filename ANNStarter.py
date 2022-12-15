import sys
import statistics
from copy import copy, deepcopy
from mlAlgo import ANNBack
from mlAlgo import extract_Data
import random
import numpy as np

# possible values for dataset_Name: [breast_Cancer | car | ecoli | letter_Recognition | mushroom]
dataset_Name = sys.argv[1]


def dataShufflingANN(data):
    sumOfAccuracy = []
    blockToShuffale = int(len(data) / 10.0)
    for i in range(10):
        print("Shuffle #: ", i)
        # 10 times
        # In each time the cross validation we need to pop from the end of the array and add to the first of the array
        for index in range(blockToShuffale):
            temp = data.pop()
            data.insert(0, temp)
        blockSize = int(len(data) / 5.0)
        DataCopy = deepcopy(data)
        for c in range(5):
            trainSet = []
            testSet = []
            # Shulffle the data by Block Size and always the 1st 1/5 data is Test and the other for the Training
            for y in range(blockSize):
                temp = DataCopy.pop()
                DataCopy.insert(0, temp)
            testSet = DataCopy[:blockSize]
            trainSet = DataCopy[blockSize + 1:]
            outputSize = len(set(extract_Data.Extract.readColumn(data, -1)))
            if outputSize <= 2:
                outputSize = 1
            ANN = ANNBack.NeuralNetwork(len(data[0]) - 1, trainSet, outputSize)
            ANN.train()
            correctMatch = 0
            # print("\n\nTest Level---\n")
            for record in range(len(testSet)):
                input_rec = deepcopy(testSet[record])
                listOutArrr = ""
                result = ANN.feedForward(input_rec)
                for out in result:
                    if out >= 0.5:
                        listOutArrr += '1'
                    else:
                        listOutArrr += '0'
                print("Target:", str(testSet[record][-1]), " Current Output: ", listOutArrr)

                if len(str(testSet[record][-1])) == 1:
                    if listOutArrr == str(testSet[record][-1]):
                       correctMatch += 1
                else:
                    if result.index(max(result)) == list(str(testSet[record][-1])).index('1'):
                        correctMatch += 1
            print('++', (correctMatch / len(testSet)) * 100)
            sumOfAccuracy.append(correctMatch / len(testSet) * 100.0)
    return sumOfAccuracy


if __name__ == '__main__':
    data = None
    extract = extract_Data.Extract()
    data = extract.readData(dataset_Name)
    # Because the target in  mushroom Data set and Letter data set is in the 1st column

    if dataset_Name == "mushroom":
        data = extract.moveTargetToLastColumn(data)
        data = extract.convertDiscreateToNumbers(data)
        # scale Data between 0 and 1
        data = extract.ScaleArray(data, list(range(22)))

    if dataset_Name == "cancer":
        data = extract.deleteColumn(data, 0)
        data = [[int(str) for str in subarray] for subarray in data]
        data = extract.ScaleArray(data, list(range(10)))
        # Convert Last column to int 1.0 to 1
        for row in data:
            row[-1] = int(row[-1])

    if dataset_Name == "car":
        data = extract.convertDiscreateToNumbers(data)
        data = extract.ScaleArray(data, list(range(6)))
        # Edit Last Column , we need to encode the values to 0 and 1
        # if we have 4 classes in output => [1000,0100,0010,0001]
        Last_Column = extract.readColumn(data, -1)
        edit_Column = []
        for value in Last_Column:
            lst = ['0'] * len(set(Last_Column))
            lst[value] = '1'
            edit_Column.append("".join([str(a) for a in lst]))
        pointer = 0
        for rec in data:
            rec[-1] = edit_Column[pointer]
            pointer += 1
        print(data)

    if dataset_Name == "letter":
        data = extract.moveTargetToLastColumn(data)
        #data = extract.convertDiscreateToNumbers(data)
        # Edit Last Column , we need to encode the values to 0 and 1
        # if we have 4 classes in output => [1000,0100,0010,0001]
        category = list(set(extract.readColumn(data, -1)))
        Output_Index = []
        Last_Column = extract.readColumn(data, -1)
        for val in Last_Column:
            Output_Index.append(category.index(val))

        edit_Column = []
        for value in Output_Index:
            lst = ['0'] * len(set(Last_Column))
            lst[value] = '1'
            edit_Column.append("".join([str(a) for a in lst]))
        pointer = 0
        for rec in data:
            rec[-1] = edit_Column[pointer]
            pointer += 1
        # data = [[int(num) for num in item] for item in data]
        for row in data:
            for c in range(len(row)-1):
                row[c] = int(row[c])
        data = extract.ScaleArray(data, list(range(16)))
        print(data)

    if dataset_Name == "ecoli":
        # Remove 1st column
        data = extract.deleteColumn(data, 0)
        category = list(set(extract.readColumn(data, -1)))
        Output_Index = []
        Last_Column = extract.readColumn(data, -1)
        for val in Last_Column:
            Output_Index.append(category.index(val))

        edit_Column = []
        for value in Output_Index:
            lst = ['0'] * len(set(Last_Column))
            lst[value] = '1'
            edit_Column.append("".join([str(a) for a in lst]))
        pointer = 0
        for rec in data:
            rec[-1] = edit_Column[pointer]
            pointer += 1
        # data = [[int(num) for num in item] for item in data]
        for row in data:
            for c in range(len(row) - 1):
                row[c] = float(row[c])
        #data = extract.ScaleArray(data, list(range(16)))
        print(data)

    random.shuffle(data)
    accuracy = dataShufflingANN(data)
    print("Accuracy: ", sum(accuracy) / 50.0)
    print("Standard Deviation: ", statistics.stdev(accuracy))
