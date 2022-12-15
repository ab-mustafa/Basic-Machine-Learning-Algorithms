import json
import math
import numpy as np


class DecisionTree:
    class Node:
        def __init__(self, data):
            self.label = data
            self.nextDec = []
            self.Desc = ""

    def __init__(self, data):
        self.map = {}
        mat1 = np.array(data)
        for i in range(len(data[0])-1):
            self.map[i] = list(set(mat1[:, i]))
        # print("MAP: ",self.map)




    def printTree(self, root, markerStr="+- ", levelMarkers=[]):
        emptyStr = " " * len(markerStr)
        connectionStr = "|" + emptyStr[:-1]
        level = len(levelMarkers)
        mapper = lambda draw: connectionStr if draw else emptyStr
        markers = "".join(map(mapper, levelMarkers[:-1]))
        markers += markerStr if level > 0 else ""
        print(f"{markers}{root.x}")
        for i, child in enumerate(root.children):
            isLast = i == len(root.children) - 1
            self.printTree(child, markerStr, [*levelMarkers, not isLast])


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

    def ID3(self, Examples, Attributes, depth=0):
        # Create a root node for the tree
        root = self.Node("")
        # If all examples are positive, Return the single-node tree Root, with label = +.
        # If all examples are negative, Return the single-node tree Root, with label = -.
        target_category = set([row[-1] for row in Examples])
        if len(target_category) == 1:
            root.label = list(target_category)[0]
            return root
        # If number of predicting attributes is empty, then Return the single node tree Root,
        # with label = most common value of the target attribute in the examples.
        if len(Attributes) == 0 or depth > 20:
            a = np.array([row[-1] for row in Examples])
            root.label = self.mostFrequent(a)
            return root

        # Otherwise Begin
        #    A ← The Attribute that best classifies examples.
        root.Desc = []
        best_Attribute_Index = -1
        best_Info_Gain = -99999999
        for Attribute in Attributes:
            Entropy = self.calculateEntropy(Examples)
            InfoGain = Entropy
            SplitData = self.splitData(Examples, Attribute)
            for key in SplitData:
                InfoGain -= ((len(SplitData[key]) / len(Examples)) * self.calculateEntropy(SplitData[key]))
            if InfoGain > best_Info_Gain:
                best_Info_Gain = InfoGain
                best_Attribute_Index = Attribute
        #    Decision Tree attribute for Root = A.
        root.label = best_Attribute_Index
        #print(SplitData)
        SplitData = self.splitData(Examples, best_Attribute_Index)
        #    For each possible value, vi, of A,
        #        Add a new tree branch below Root, corresponding to the test A = vi.
        #        Let Examples(vi) be the subset of examples that have the value vi for A
        #        If Examples(vi) is empty
        #            Then below this new branch add a leaf node with label = most common target value in the examples
        #        Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
        # End
        newAttrib = Attributes[:]
        newAttrib.remove(best_Attribute_Index)
        for key in SplitData:
            if len(SplitData[key]) == 0:
                leaf = self.Node("")
                a = np.array([row[-1] for row in Examples])
                leaf.label = self.mostFrequent(a)
                leaf.nextDec = []
                root.Desc.append(key)
                root.nextDec.append(leaf)
            else:
                root.Desc.append(key)
                root.nextDec.append(self.ID3(SplitData[key], newAttrib, depth+1))
        # Return Root
        return root

    #Code Copied
    def printTree(self, root, level=0):
        print("  " * level, root.label, root.Desc)
        for child in root.nextDec:
            self.printTree(child, level + 1)

    def extractFromTree(self, tree, input):
        # Example
        if len(tree.nextDec) == 0:
            # Return the value of label
            return tree.label
        # from the input get the value related to label like Temp "low"
        value = input[tree.label]
        # get the index of the value
        value = tree.Desc.index(value)
        return self.extractFromTree(tree.nextDec[value], input)


"""
DT = DecisionTree([["Sunny", "Hot", "High", "Weak", "No"],
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
                       ])


DecisionTree = DT.ID3([["Sunny", "Hot", "High", "Weak", "No"],
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
                       ],
                      [0, 1, 2, 3])

print("\n------------------\n\n")
DT.printTree(DecisionTree)
result = DT.extractFromTree(DecisionTree, ["Sunny", "Mild", "Normal", "Weak"])
print("\n", result)

"""