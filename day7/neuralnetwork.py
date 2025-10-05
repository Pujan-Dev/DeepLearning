import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Activation function (Sigmoid)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward pass
    def feedforward(self, X):
        # Input to hidden
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden to output
        self.output_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.predicted_output = self.sigmoid(self.output_input)

        return self.predicted_output

    # Backward pass (Backpropagation)
    def backward(self, X, y, learning_rate):
        # Error in output
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Error in hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += (
            np.dot(self.hidden_output.T, output_delta) * learning_rate
        )
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Training function
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backward(X, y, learning_rate)

            if epoch % 1000 == 0:  # print every 1000 epochs
                loss = np.mean(np.square(y - self.predicted_output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Example usage
if __name__ == "__main__":
    # XOR Problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    print("Final predictions:")
    print(nn.feedforward(X))
