import sys
import statistics
from copy import copy, deepcopy
from mlAlgo import NaiveBayes
from mlAlgo import extract_Data
import random
from mlAlgo import ANNBack as bNN
from mlAlgo.AdaBoost import AdaBoost

# possible values for dataset_Name: [breast_Cancer | car | ecoli | letter_Recognition | mushroom]
dataset_Name = sys.argv[1]


def dataShufflingDT(data):
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
            ABoost = AdaBoost(data, trainSet)
            # check the Accuracy
            correctMatch = 0
            for record in range(len(testSet)):
                input_rec = deepcopy(testSet[record])
                input_rec.pop()
                # print("Input Recored" , input_rec)
                result = ABoost.extractFromModel(input_rec)
                print("Input record ", testSet[record], " : ", result)
                if result == testSet[record][-1]:
                    correctMatch += 1
            print('++', (correctMatch / len(testSet)) * 100)
            sumOfAccuracy.append(correctMatch / len(testSet) * 100.0)
    return sumOfAccuracy


def ConvertRowToDiscreate(InputRow, categoryInterval=5):
    Row = [float(x) for x in InputRow]
    maxValue = max(Row)
    minValue = min(Row)
    # we will always divide it to 5 category
    distance = (maxValue - minValue) / categoryInterval
    print("Distance",distance)
    ruler = [minValue]
    while ruler[-1] < maxValue:
        ruler.append(ruler[-1] + distance)
    print("Ruler:", ruler)
    discreateArray = []
    for val in Row:
        # Check Ruler to find the interval
        for i in range(len(ruler)):
            if i == len(ruler) - 1:
                discreateArray.append(str(i - 1))
                break
            elif ruler[i] <= val < ruler[i + 1]:
                discreateArray.append(str(i))
                break
    # Check the length
    if len(Row) != len(discreateArray):
        raise Exception("Error in Converting Numeric value to categorial values")
    return discreateArray


def SaveColumn(OriginArray, ColumnData, index):
    count = 0
    for value in ColumnData:
        OriginArray[count][index] = value
        count += 1


if __name__ == '__main__':
    # working data set: car,mushroom,ecoli
    data = None
    extract = extract_Data.Extract()
    data = extract.readData(dataset_Name)
    # Because the target in  mushroom Data set and Letter data set is in the 1st column
    if dataset_Name == "mushroom":
        data = extract.moveTargetToLastColumn(data)

    if dataset_Name == "letter":
        data = extract.moveTargetToLastColumn(data)
        # for each column convert continuous to Discrete :5,6
        columnIndexToChange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for index in columnIndexToChange:
            array = extract.readColumn(data, index)
            Discreate_array = ConvertRowToDiscreate(array, 20)
            SaveColumn(data, Discreate_array, index)

    if dataset_Name == "ecoli":
        # Remove 1st column
        data = extract.deleteColumn(data, 0)
        # for each column convert continuous to Discrete :5,6
        columnIndexToChange = [0, 1, 2, 3, 4, 5, 6]
        for index in columnIndexToChange:
            array = extract.readColumn(data, index)
            Discreate_array = ConvertRowToDiscreate(array)
            SaveColumn(data, Discreate_array, index)


    if dataset_Name == "cancer":
        data = extract.deleteColumn(data, 0)
        # for each column convert continuous to Discrete :5,6
        columnIndexToChange = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for index in columnIndexToChange:
            array = extract.readColumn(data, index)
            Discreate_array = ConvertRowToDiscreate(array)
            SaveColumn(data, Discreate_array, index)
    random.shuffle(data)
    accuracy = dataShufflingDT(data)
    print("Accuracy: ", sum(accuracy) / 50.0)
    print("Standard Deviation: ", statistics.stdev(accuracy))



