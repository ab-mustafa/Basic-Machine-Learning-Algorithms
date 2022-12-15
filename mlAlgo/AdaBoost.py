import json
import math
import numpy as np


class AdaBoost:
    class StumpTree:
        def __init__(self, featureIndex):
            self.featureIndex = featureIndex
            self.featureValues = []
            self.Outputs = []
            self.alphaError = 0
            self.InfoGain = 0

    def __init__(self, data, trainData):
        self.stumpTrees = []  # to save all the trees
        self.map = {}
        # store all possible values for each column (input & output)
        mat1 = np.array(data)
        for i in range(len(data[0]) - 1):
            self.map[i] = list(set(mat1[:, i]))
        self.outputDiscreate = list(set(mat1[:, -1]))
        # print("MAP: ",self.map)
        # Output: MAP: {0: ['Sunny', 'Rain', 'Overcast'], 1: ['Cool', 'Hot', 'Mild'], ...}
        # Create stump Tree for each feature
        self.generateStumpTrees(trainData)
        # Intialize weights
        self.weights = [1/len(trainData) for i in range(len(trainData))]
        self.triggerBoosting(trainData)
        print("Trees:")
        self.printStumpTrees()

    def calculateEntropy(self, data):
        # Assuming Data 2D Array
        # Assuming target value index is last column
        target_Values = [row[-1] for row in data]
        dictionary = {}
        for value in target_Values:
            dictionary[value] = dictionary.get(value, 0) + 1
        numberOfOutput = sum(dictionary.values())
        entropy = 0
        for key in dictionary:
            entropy += (-1 * (dictionary[key] / numberOfOutput) * math.log2(dictionary[key] / numberOfOutput))
        # print(dictionary)
        return entropy

    def splitData(self, data, column_index):
        # Get the Target values for this column
        target_Values = [row[column_index] for row in data]
        # Create Dict to save the index
        dict = {}
        for index in range(len(target_Values)):
            key = target_Values[index]
            if key in dict:
                dict[key].append(data[index])
            else:
                dict[key] = [data[index]]
        # Add Empty Vi to dict
        for VI in self.map[column_index]:
            if VI not in dict.keys():
                dict[VI] = []
        # print("\n\n++++++++++++++++++",dict)
        return dict

    def mostFrequent(self, arr):
        n = len(arr)
        maxcount = 0
        element_having_max_freq = 0
        for i in range(0, n):
            count = 0
            for j in range(0, n):
                if arr[i] == arr[j]:
                    count += 1
            if count > maxcount:
                maxcount = count
                element_having_max_freq = arr[i]
        return element_having_max_freq

    def extractFromStumpTree(self, tree, input):
        index = int(tree.featureIndex)
        valueF = input[index]
        return tree.Outputs[tree.featureValues.index(valueF)]

    def triggerBoosting(self, Examples):
        for R in range(len(Examples[0]) - 1):
            # Normalize weights
            sumOfWights = sum(self.weights)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] / sumOfWights
            # get weak Learner
            weakLearner = self.stumpTrees[R]
            # Compute Err
            Err = 0
            for rowN in range(len(Examples)):
                actualOutput = self.extractFromStumpTree(weakLearner, Examples[rowN])
                if actualOutput != Examples[rowN][-1]:
                    Err += (self.weights[rowN] * 1.0)
            # compute Alpha error SAMME
            alphaErr = math.log10((1-Err)/Err) + math.log10(len(self.outputDiscreate)-1)
            weakLearner.alphaError = alphaErr
            # Update Weights
            for rowN in range(len(Examples)):
                actualOutput = self.extractFromStumpTree(weakLearner, Examples[rowN])
                if actualOutput == Examples[rowN][-1]:
                    self.weights[rowN] *= math.pow(math.e, -1*alphaErr)
                else:
                    self.weights[rowN] *= math.pow(math.e, 1 * alphaErr)

    def printStumpTrees(self):
        for i in range(len(self.stumpTrees)):
            print("\n")
            print("Feature: ", self.stumpTrees[i].featureIndex)
            print("Value: ", self.stumpTrees[i].featureValues)
            print("Outputs: ", self.stumpTrees[i].Outputs)
            print("Info Gain: ", self.stumpTrees[i].InfoGain)
            print("Info alphaError: ", self.stumpTrees[i].alphaError)

    def generateStumpTrees(self, Examples):
        Entropy = self.calculateEntropy(Examples)
        # For Each feature generate stump tree (Weak Classifier)
        for T in range(len(Examples[0]) - 1):
            # Create a root node for the tree
            root = self.StumpTree(str(T))
            SplitData = self.splitData(Examples, T)
            # print("\n Index: ", T, "  Split Data: ", SplitData)
            InfoGain = Entropy
            for key in SplitData:
                InfoGain -= ((len(SplitData[key]) / len(Examples)) * self.calculateEntropy(SplitData[key]))
                root.featureValues.append(key)
                list = np.array([row[-1] for row in SplitData[key]])
                root.Outputs.append(self.mostFrequent(list))
            root.InfoGain = InfoGain
            self.stumpTrees.append(root)
            # Sort the trees depends on Info Gain
            self.stumpTrees = sorted(self.stumpTrees, key=lambda x: x.InfoGain, reverse=True)

    def extractFromModel(self, input):
        outputStructure = {}
        for val in self.outputDiscreate:
            outputStructure[val] = 0
        for val in self.outputDiscreate:
            for tree in self.stumpTrees:
                if self.extractFromStumpTree(tree, input) == val:
                    outputStructure[val] = outputStructure.get(val) + tree.alphaError
        # print(outputStructure)
        return sorted(outputStructure, key=lambda x: outputStructure[x])[-1]






"""
Ex = [["Sunny", "Hot", "High", "Weak", "No"],
      ["Sunny", "Hot", "High", "Strong", "No"],
      ["Overcast", "Hot", "High", "Weak", "Yes"],
      ["Rain", "Mild", "High", "Weak", "Yes"],
      ["Rain", "Cool", "Normal", "Weak", "Yes"],
      ["Rain", "Cool", "Normal", "Strong", "No"],
      ["Overcast", "Cool", "Normal", "Strong", "Yes"],
      ["Sunny", "Mild", "High", "Weak", "No"],
      ["Sunny", "Cool", "Normal", "Weak", "Yes"],
      ["Rain", "Mild", "Normal", "Weak", "Yes"],
      ["Sunny", "Mild", "Normal", "Strong", "Yes"],
      ["Overcast", "Mild", "High", "Strong", "Yes"],
      ["Overcast", "Hot", "Normal", "Weak", "Yes"],
      ["Rain", "Mild", "High", "Strong", "No"]
      ]

DT = AdaBoost(Ex, Ex)
DT.printStumpTrees()
for row in Ex:
    print(row[-1], DT.extractFromModel(row))

"""
