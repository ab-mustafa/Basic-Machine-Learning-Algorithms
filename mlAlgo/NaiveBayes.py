from copy import copy, deepcopy


class NaiveBayes:
    def __init__(self, data):
        self.outputMap = {}
        self.attributesMap = {}
        # Split Input and output
        output_Array = []
        input_Array = []
        for row in data:
            vector = deepcopy(row)
            output_Array.append(vector.pop())
            input_Array.append(vector)
        # print("Input Array:", input_Array)
        # print("Output Array:", output_Array)
        self.train(input_Array, output_Array)

    def readColumn(self, data, index):
        s = []
        for row in data:
            s.append(row[index])
        return s

    def train(self, inputAttributes, output):
        # Validation
        if len(output) != len(inputAttributes):
            raise Exception("Wrong Params , Check Prams")
        # Discover output
        s = list(set(output))
        # Count the output
        for category in output:
            self.outputMap[category] = {"Count": self.outputMap.get(category, {"Count": 0}).get("Count") + 1}
        for category in self.outputMap:
            self.outputMap[category]["Probability"] = self.outputMap[category]["Count"] / len(output)
        # Define Json for attributes
        #  Init Stage
        Number_Of_Columns = len(inputAttributes[0])
        for column in range(Number_Of_Columns):
            categories = set(self.readColumn(inputAttributes, column))
            self.attributesMap["COLUMN" + str(column)] = {}
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value] = {}
                for output_Category in self.outputMap:
                    self.attributesMap["COLUMN" + str(column)][value][output_Category] = 0
        #  Fill Stage
        for column in range(Number_Of_Columns):
            categories = self.readColumn(inputAttributes, column)
            POINTER = 0
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value][output[POINTER]] += (
                            1 / self.outputMap[output[POINTER]]["Count"])
                POINTER += 1

    def examine(self, vector):
        examine_result = {}
        for category in self.outputMap:
            examine_result[category] = 1

        # Calculate Prob for each Output
        for category in self.outputMap:
            examine_result[category] *= self.outputMap[category]["Probability"]
            for c in range(len(vector)):
                try:
                    examine_result[category] *= self.attributesMap["COLUMN"+str(c)][vector[c]][category]
                except KeyError:
                    examine_result[category] *= 0
        result = sorted(examine_result, key=lambda x: examine_result[x])[-1]
        return result

NB = NaiveBayes([["Sunny", "Hot", "High", "Weak", "No"],
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
print("Output:", NB.outputMap)
print("Attribute MAP:", NB.attributesMap)
NB.examine(["Sunny", "Cool", "High", "Strong"])
