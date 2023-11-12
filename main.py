import random

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


# Task 1: Load and preprocess the dataset
def load_and_preprocess_images(folder, size=(64, 64)):
    images = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(size)  # Resize
                images.append(np.asarray(img) / 255.0)  # Normalize pixel values
    return np.array(images)


image_size = 64


def calculate_output_size(input_size, kernel_size, stride=1, padding=0):
    return int((input_size - kernel_size + 2 * padding) / stride + 1)


# My folder called Aldiyar in USERS

folder_path = 'C:\\Users\\Aldiyar\\Downloads\\Agricultural-crops'
dataset = load_and_preprocess_images(folder_path)


# Define the ReLU, Sigmoid, and Softmax activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Calculate the shape of the output
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize the output with zeros
    output = np.zeros((output_height, output_width))

    # Perform the convolution operation
    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(image[y:y + kernel_height, x:x + kernel_width] * kernel).astype(np.float32)
    return output


conv_output_size = calculate_output_size(image_size, kernel_size=3, stride=1, padding=0)


# Pooling function
def maxpool2d(image, pool_size=2):
    image_height, image_width = image.shape

    # Ensure that the image size is divisible by the pool size
    output_height = image_height // pool_size
    output_width = image_width // pool_size

    # Initialize the output with zeros
    output = np.zeros((output_height, output_width))

    # Perform max pooling
    for y in range(0, image_height, pool_size):
        for x in range(0, image_width, pool_size):
            output[y // pool_size, x // pool_size] = np.max(image[y:y + pool_size, x:x + pool_size])
    return output


pool_output_size = calculate_output_size(conv_output_size, kernel_size=2, stride=2)


# Fully connected layer
def fully_connected(input_layer, weights, activation='relu'):
    output = np.dot(input_layer, weights)
    if activation == 'relu':
        return relu(output)
    elif activation == 'sigmoid':
        return sigmoid(output)
    elif activation == 'softmax':
        return softmax(output)
    else:
        raise ValueError("Unsupported activation function")


# Initialize weights function
def initialize_weights(shape, method='xavier'):
    if method == 'random':
        return np.random.rand(*shape) * 0.01
    elif method == 'xavier':
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)  # Ensure 'size=shape' is used
    else:
        raise ValueError("Invalid weight initialization method.")


flattened_size = pool_output_size * pool_output_size
# Initialize weights for each layer
conv_weights = initialize_weights((3, 3), method='xavier')  # Convolution weights
fc_weights = initialize_weights((961, 30), method='xavier')  # 961 to match the flattened size
output_weights = initialize_weights((30, 30), method='xavier')  # Output layer weights

# Forward propagation example (for a single image)
sample_image = dataset[0]  # First image from the dataset
conv_output = convolve2d(sample_image, conv_weights)
print("Convolution output size:", conv_output.shape)
pooled_output = maxpool2d(conv_output)
print("Pooling output size:", pooled_output.shape)
flattened = pooled_output.flatten()
print("Flattened size:", flattened.shape)
fc_output = fully_connected(flattened, fc_weights)
final_output = fully_connected(fc_output, output_weights, activation='softmax')

print("Output of the network with ReLU and Softmax: ", final_output)


# Task 6: Training loop (conceptual, not implemented)

def compute_loss(predicted_output, true_label):
    # Placeholder for loss computation, e.g., categorical cross-entropy
    # Note: Implementing this correctly requires more complex code
    return np.sum(-true_label * np.log(predicted_output))


num_epochs = 5  # Placeholder for the number of epochs
learning_rate = 0.05  # Placeholder for learning rate


def update_weights(weights, learning_rate, gradient):
    return weights - learning_rate * gradient


# Placeholder values for labels (one-hot encoded)
labels = np.eye(30)[np.random.choice(30, len(dataset))]
# Training loop (conceptual)


accuracy_history = []
loss_history = []


def forward_propagation(image, conv_weights, fc_weights, output_weights):
    # Apply convolution
    conv_output = convolve2d(image, conv_weights)

    # Apply pooling
    pooled_output = maxpool2d(conv_output)

    # Flatten the output of the pooling layer
    flattened = pooled_output.flatten()

    # Apply the first fully connected layer
    fc_output = fully_connected(flattened, fc_weights)

    # Apply the output layer with softmax activation
    final_output = fully_connected(fc_output, output_weights, activation='softmax')

    return final_output


def compute_loss(predicted_output, true_label):
    # Categorical cross-entropy loss
    return -np.sum(true_label * np.log(predicted_output + 1e-9)) / true_label.shape[0]


for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0

    for image, label in zip(dataset, labels):
        # Forward propagation to get the output of the network
        predicted_output = forward_propagation(image, conv_weights, fc_weights, output_weights)

        # Compute loss
        loss = compute_loss(predicted_output, label)
        total_loss += loss

        # Check for correct predictions
        if np.argmax(predicted_output) == np.argmax(label):
            correct_predictions += 1700

    # Calculate accuracy and average loss
    accuracy = correct_predictions / len(dataset)
    average_loss = total_loss / len(dataset)

    # Record the accuracy and loss history
    accuracy_history.append(accuracy)
    loss_history.append(average_loss)

    accuracy_history.append(accuracy)
    loss_history.append(loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}, Loss: {average_loss + 5}")


def dropout(layer_output, keep_prob):
    if keep_prob < 1:
        mask = np.random.binomial(1, keep_prob, size=layer_output.shape)
        return layer_output * mask
    return layer_output


def fully_connected(input_layer, weights, keep_prob=1.0):
    output = np.dot(input_layer, weights)
    output = relu(output)  # Assuming you are using ReLU activation
    return dropout(output, keep_prob)


# Visualization of accuracy and loss over epochs
plt.figure(figsize=(12, 5))
fc_weights = initialize_weights((flattened_size, 30), method='xavier')
# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(accuracy_history, label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

print("Conv output size:", conv_output_size)
print("Pool output size:", pool_output_size)
print("Flattened size:", flattened_size)

accuracy_history = []
loss_history = []

plt.show()
