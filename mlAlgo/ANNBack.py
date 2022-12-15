import sys
import numpy as np
import math
from copy import copy, deepcopy


class NeuralNetwork:

    def multiplyArray(self, arr1, arr2):
        if len(arr1) != len(arr2):
            raise Exception("the length values for the two array not equal")
        prod = []
        for element in range(len(arr1)):
            prod.append((arr1[element] * arr2[element]))
        return prod

    def scaleArray(self, min=-1, max=1):
        for r in range(len(self.W1)):
            for c in range(len(self.W1[0])):
                self.W1[r][c] = self.minmax(1, 0, 1, -1, self.W1[r][c])
        for r in range(len(self.W2)):
            for c in range(len(self.W2[0])):
                self.W2[r][c] = self.minmax(1, 0, max, min, self.W2[r][c])
        for r in range(len(self.W3)):
            for c in range(len(self.W3[0])):
                self.W3[r][c] = self.minmax(1, 0, max, min, self.W3[r][c])

    def __init__(self, inputSize, trainData, outputSize=1):
        # parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = 3
        self.hiddenSize2 = 5
        self.LearningRate = 0.2
        self.trainData = trainData
        self.MaxNumberEpochIteration = 3000
        self.MaxErrorPerEpoch = 10
        # Row = Node
        # Columns Weight
        self.W1 = np.random.random((self.hiddenSize1, 1 + inputSize))  # (3,4) weight matrix from input to hidden layer
        self.W2 = np.random.random(
            (self.hiddenSize2, 1 + self.hiddenSize1))  # (3,4) weight matrix from input to hidden layer
        self.W3 = np.random.random(
            (self.outputSize, self.hiddenSize2 + 1))  # (1x4) weight matrix from hidden to output layer

        # self.W1 = [ [0.66989521,0.06426627,0.35890368],[0.68983257,0.23465994,0.49512472]]
        # self.W2 = [ [0.5975942,0.11017504,0.19100509], [0.84362423,0.56460102,0.59309509]]
        # self.W3 = [ [0.83439889,0.60226426,0.90141674]]

        print("W1", self.W1)
        print("W2", self.W2)
        print("W3", self.W3)

        self.layer1Out = []
        self.layer2Out = []
        self.layer3Out = []
        # self.scaleArray()

    def feedForward(self, X):
        inputData = deepcopy(X)
        # Remove Actual Input
        inputData.pop()
        # forward propogation through the network
        self.layer1Out = []
        inputVector = deepcopy(inputData)
        # Add bias for input Vector
        inputVector.insert(0, 1)
        for node in self.W1:
            result = self.sigmoid(sum(self.multiplyArray(node, inputVector)))
            self.layer1Out.append(result)

        self.layer2Out = []
        inputVector = deepcopy(self.layer1Out)
        inputVector.insert(0, 1)
        for node in self.W2:
            result = self.sigmoid(sum(self.multiplyArray(node, inputVector)))
            self.layer2Out.append(result)

        self.layer3Out = []
        inputVector = deepcopy(self.layer2Out)
        inputVector.insert(0, 1)
        for node in self.W3:
            result = self.sigmoid(sum(self.multiplyArray(node, inputVector)))
            self.layer3Out.append(result)

        # print("Layer 1:", self.layer1Out)
        # print("Layer 2:", self.layer2Out)
        # print("Layer 3:", self.layer3Out)
        return self.layer3Out

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def relu(self, s, deriv=False):
        if (deriv == True):
            if s > 0:
                return 1
            else:
                return 0
        if s > 0:
            return s
        else:
            return 0

    def minmax(self, N_max, N_min, I_max, I_min, value):
        return ((value - I_min) * ((N_max - N_min) / (I_max - I_min))) + N_min

    def CheckIfEqualSize(self, Arr1, Arr2):
        if len(Arr1) == len(Arr2) and len(Arr1[0]) == len(Arr2[0]):
            return True
        else:
            return False

    def train(self, multiClass=False):
        # Batch Mode
        EpochNumber = 0
        BatchCountErr = 0
        BatchNextFlag = True
        while BatchNextFlag:
            EpochNumber += 1
            if EpochNumber >= self.MaxNumberEpochIteration:
                break
            print("EPOCH #: ", EpochNumber)
            for record in self.trainData:
                vector = deepcopy(record)
                target = list(str(vector[-1]))
                target = list(map(int, target))
                output = self.feedForward(vector)

                # print("Output", output)
                # print("TARGET", target)

                ErrorArray = []
                AbsoluteOutputArray = []
                for m in range(len(output)):
                    ErrorArray.append(target[m] - output[m])
                    if output[m] >= 0.5:
                        AbsoluteOutputArray.append('1')
                    else:
                        AbsoluteOutputArray.append('0')

                # print("Error", ErrorArray)
                # print("Absolute Error", AbsoluteErrorArray)

                # Skip Update if all the value is correct [No Error In output]
                A = str(vector[-1])
                B = "".join([str(elem) for elem in AbsoluteOutputArray])
                if A == B:
                    continue
                BatchCountErr += 1
                # Update Weights FOR w3
                W3_diff = []
                W3_gradient = []
                CX = 0
                for node in self.W3:
                    temp = []
                    gradient_Error = output[CX] * (1 - output[CX]) * ErrorArray[CX]
                    W3_gradient.append(gradient_Error)
                    for weight in range(len(node)):
                        if weight == 0:
                            diff = self.LearningRate * 1 * gradient_Error
                        else:
                            diff = self.LearningRate * self.layer2Out[weight - 1] * gradient_Error
                        temp.append(diff)
                    W3_diff.append(temp)
                    CX += 1

                # Update Weights FOR w2
                W2_diff = []
                W2_gradient = []
                F = 0
                for node in self.W2:
                    temp = []
                    subgrad = 0
                    CX = 0
                    for b in range(len(self.W3)):
                        subgrad += self.W3[b][F + 1] * W3_gradient[CX]
                        CX += 1
                    gradient_Error = self.layer2Out[F] * (1 - self.layer2Out[F]) * subgrad
                    W2_gradient.append(gradient_Error)
                    for weight in range(len(node)):
                        if weight == 0:
                            diff = self.LearningRate * 1 * gradient_Error
                        else:
                            diff = self.LearningRate * self.layer1Out[weight - 1] * gradient_Error
                        temp.append(diff)
                    W2_diff.append(temp)
                    F += 1
                # Update Weights FOR w1
                W1_diff = []
                W1_gradient = []
                F = 0
                for node in self.W1:
                    temp = []
                    subgrad = 0
                    CX = 0
                    for n in range(len(self.W2)):
                        subgrad += self.W2[n][F + 1] * W2_gradient[CX]
                        CX += 1
                    gradient_Error = self.layer1Out[F] * (1 - self.layer1Out[F]) * subgrad
                    W1_gradient.append(gradient_Error)
                    for weight in range(len(node)):
                        if weight == 0:
                            diff = self.LearningRate * 1 * gradient_Error
                        else:
                            diff = self.LearningRate * vector[weight - 1] * gradient_Error
                        temp.append(diff)
                    W1_diff.append(temp)
                    F += 1

                # Check if the Updated
                # print("\n\n\n---------DIFF------------\n\n\n")
                # print("W1", W1_diff)
                # print("W2", W2_diff)
                # print("W3", W3_diff)
                # sys.exit(0)
                if not (self.CheckIfEqualSize(W1_diff, self.W1) and self.CheckIfEqualSize(W2_diff, self.W2)
                        and self.CheckIfEqualSize(W3_diff, self.W3)):
                    raise Exception("Wrong Error Updates")
                # Change Weights
                for r in range(len(self.W1)):
                    for c in range(len(self.W1[0])):
                        self.W1[r][c] += W1_diff[r][c]
                for r in range(len(self.W2)):
                    for c in range(len(self.W2[0])):
                        self.W2[r][c] += W2_diff[r][c]
                for r in range(len(self.W3)):
                    for c in range(len(self.W3[0])):
                        self.W3[r][c] += W3_diff[r][c]
                # Check if the Updated
                # print("---------After Updated------------")
                # print("W1", self.W1)
                # print("W2", self.W2)
                # print("W3", self.W3)

            if BatchCountErr <= self.MaxErrorPerEpoch:
                print("Epoch Number", EpochNumber)
                BatchNextFlag = False
            else:
                BatchCountErr = 0
                continue

# NN = NeuralNetwork(2, [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
# NN.train()
# print("------------Test--------------------\n")
# print(NN.feedForward([0, 0, 0]))
# print(NN.feedForward([0, 1, 0]))
# print(NN.feedForward([1, 0, 0]))
# print(NN.feedForward([1, 1, 1]))
