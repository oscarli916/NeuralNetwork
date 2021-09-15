import random
import math

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []

        for i in range(rows):
            self.matrix.append([])
            for _ in range(cols):
                self.matrix[i].append(0)


    def randomize(self, f, l):
        for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] = random.uniform(f, l)
    

    @staticmethod
    def mulitply(m, n):
        if m.cols == n.rows:
            # Matrix product
            result = Matrix(m.rows, n.cols)
            a = m.matrix
            b = n.matrix
            for i in range(result.rows):
                for j in range(result.cols):
                    # Dot product
                    sum = 0
                    for k in range(m.cols):
                        sum += a[i][k] * b[k][j]
                    result.matrix[i][j] = sum 
            return result

    # This can delete m, but i keep it first
    def multiple(self, m, n):
        if type(n) == Matrix:
            # Element-wise product
            for i in range(m.rows):
                for j in range(m.cols):
                    m.matrix[i][j] *= n.matrix[i][j]

        else:
            # Scalor product
            for i in range(m.rows):
                for j in range(m.cols):
                    m.matrix[i][j] *= n


    # Add can delete m argument, but i keep it first
    def add(self, m, n):
        if type(n) == Matrix:
            # Element-wise sum
            for i in range(m.rows):
                for j in range(m.cols):
                    m.matrix[i][j] += n.matrix[i][j]
        else:
            # Scalor sum
            for i in range(m.rows):
                for j in range(m.cols):
                    m.matrix[i][j] += n


    @staticmethod
    def subtract(a,b):
        # Return a new Matrix a-b
        result = Matrix(a.rows, a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j]
        return result


    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for i in range(result.rows):
            for j in range(result.cols):
                result.matrix[i][j] = m.matrix[j][i]     
        return result


    # def transpose(self):
    #     result = Matrix(self.cols, self.rows)
    #     for i in range(result.rows):
    #         for j in range(result.cols):
    #             result.matrix[i][j] = self.matrix[j][i]     
    #     return result               


    @staticmethod
    def mad(m, func):
        result = Matrix(m.rows, m.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                val = m.matrix[i][j]
                result.matrix[i][j] = func(val)
        return result
        


    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.matrix[i][j]
                self.matrix[i][j] = func(val)

    
    @staticmethod
    def fromarray(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.matrix[i][0] = arr[i]
        return m


    def toarray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.matrix[i][j])
        return arr

    
    def show(self):
        print(self.matrix)


    def show_neat(self):
        # Print matrix neatly
        for i in self.matrix:
            print(i)



class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weight_ih = Matrix(hidden_nodes, input_nodes)
        self.weight_ho = Matrix(output_nodes, hidden_nodes)
        self.weight_ih.randomize(-1, 1)
        self.weight_ho.randomize(-1, 1)

        self.bias_ih = Matrix(hidden_nodes, 1)
        self.bias_ho = Matrix(output_nodes, 1)
        self.bias_ih.randomize(0, 1)
        self.bias_ho.randomize(0, 1)

        self.learning_rate = 0.3


    def feedforward(self, input_array):

        # Input -> Hidden

        # Input array to Matrix
        inputs = Matrix.fromarray(input_array)
        # Multiply weights to input
        hidden = Matrix.mulitply(self.weight_ih, inputs)
        # Add bias to result
        hidden.add(hidden, self.bias_ih)
        # Activation
        hidden.map(Sigmoid)


        # Hidden -> Output

        # Multiply weights to input
        output = Matrix.mulitply(self.weight_ho, hidden)
        # Add bias to result
        output.add(output, self.bias_ho)
        # Activation
        output.map(Sigmoid)
        

        return output.toarray()

    
    def train(self, input_array, target_array):
        # Input -> Hidden

        # Input array to Matrix
        inputs = Matrix.fromarray(input_array)
        # Multiply weights to input
        hidden = Matrix.mulitply(self.weight_ih, inputs)
        # Add bias to result
        hidden.add(hidden, self.bias_ih)
        # Activation
        hidden.map(Sigmoid)


        # Hidden -> Output

        # Multiply weights to input
        output = Matrix.mulitply(self.weight_ho, hidden)
        # Add bias to result
        output.add(output, self.bias_ho)
        # Activation
        output.map(Sigmoid)
        

        # Backpropagation


        # Convert array to Matrix
        target = Matrix.fromarray(target_array)
        # Calculate the output error
        output_error = Matrix.subtract(target ,output)

        # Calculate the gradient
        gradient = Matrix.mad(output, dSigmoid)
        gradient.multiple(gradient, output_error)
        gradient.multiple(gradient, self.learning_rate)

        # Calculate H->O delta
        hidden_t = Matrix.transpose(hidden)
        weight_ho_delta = Matrix.mulitply(gradient, hidden_t)


        # Change the weight_ho
        self.weight_ho.add(self.weight_ho, weight_ho_delta)
        # Change the bias_ho
        self.bias_ho.add(self.bias_ho, gradient)




        # Calculate the hidden error

        # Transpose the weight_ho
        weight_ho_t = Matrix.transpose(self.weight_ho)
        hidden_error = Matrix.mulitply(weight_ho_t, output_error)

        # Calculate the hidden_gradient
        hidden_gradient = Matrix.mad(hidden, dSigmoid)
        hidden_gradient.multiple(hidden_gradient, hidden_error)
        hidden_gradient.multiple(hidden_gradient, self.learning_rate)

        # Calculate I->H delta
        input_t = Matrix.transpose(inputs)
        weight_ih_delta = Matrix.mulitply(hidden_gradient, input_t)

        # Change the weight_ih
        self.weight_ih.add(self.weight_ih, weight_ih_delta)
        # Change the bias_ih
        self.bias_ih.add(self.bias_ih, hidden_gradient)


def Sigmoid(x):
    return 1/(1 + math.exp(-x))

def dSigmoid(x):
    return x * (1 - x)


if __name__ == "__main__":

    nn = NeuralNetwork(2,3,1) # Hidden layer have to choose wisely


    training_data = [
        {"inputs":[0,0],
        "targets":[0]},
        {"inputs":[0,1],
        "targets":[1]},
        {"inputs":[1,0],
        "targets":[1]},
        {"inputs":[1,1],
        "targets":[0]}
    ]
    # XOR



    for _ in range(10000):
        r = random.randint(0,3)
        nn.train(training_data[r]["inputs"], training_data[r]["targets"])


    guess = nn.feedforward([0,0])
    print("Guess for 0,0: " + str(guess))
    guess = nn.feedforward([0,1])
    print("Guess for 0,1: " + str(guess))
    guess = nn.feedforward([1,0])
    print("Guess for 1,0: " + str(guess))
    guess = nn.feedforward([1,1])
    print("Guess for 1,1: " + str(guess))
