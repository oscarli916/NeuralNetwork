from NeuralNetwork import Matrix
import random

inputs = [[1,1,1],
          [2,6,3],
          [4,8,8],
          [7,4,9],
          [3,7,4],
          [5,3,2],
          [11,2,7],
          [9,5,6],
          [8,10,5],
          [6,13,12],
          [12,9,11],
          [13,11,14],
          [10,12,13],
          [14,14,10]]

variable = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]


# print(inputs[0])
# print(variable)
result = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
while True:
    check = []
    for i in range(len(inputs)):
        answer = 0
        for j in range(len(inputs[i])):
            temp = inputs[i][j] * variable[j]
            answer = answer + temp
        # print(answer)
        check.append(answer)
    # print(check)
    for k in range(len(check)):
        wrong = 0
        if result[k] - 0.5 <= check[k] <= result[k] + 0.5:
            pass
        else: 
            wrong = 1
    if wrong == 0:
        break
print(variable)