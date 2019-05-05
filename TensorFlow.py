### Introduction to TensorFlow in Python

### 1. Introduction to TensorFlow

# Define a 3x4 tensor with all values equal to 9
x = fill([3, 4], 9)

# Define a tensor of ones with the same shape as x
y = ones_like(x)

# Define the one-dimensional vector, z
z = constant([1, 2, 3, 4])

# Print z as a numpy array
print(z.numpy())

# Define the 1-dimensional variable X
X = Variable([1, 2, 3, 4])

# Print the variable X
print(X)

# Convert X to a numpy array and assign it to Z
Z = X.numpy()

# Print the numpy array Z
print(Z)

# Define tensors A0 and B0 as constants
A0 = constant([1, 2, 3, 4])
B0 = constant([[1, 2, 3], [1, 6, 4]])

# Define A1 and B1 to have the correct shape
A1 = ones_like(A0)
B1 = ones_like(B0)

# Perform element-wise multiplication
A2 = multiply(A0, A1)
B2 = multiply(B0, B1)

# Print the tensors A2 and B2
print(A2.numpy())
print(B2.numpy())

# Define X, b, and y as constants
X = constant([[1, 2], [2, 1], [5, 8], [6, 10]])
b = constant([[1], [2]])
y = constant([[6], [4], [20], [23]])

# Compute ypred using X and b
ypred = matmul(X, b)

# Compute the error as y - ypred
error = subtract(y, ypred)

# Define input data
image = ones([16, 16])

# Reshape input data into a vector
image_vector = reshape(image, ([256, 1]))

# Reshape input data into a higher dimensional tensor
image_tensor = reshape(image, ([4, 4, 4, 4]))

# Define input data
image = ones([16, 16, 3])

# Reshape input data into a vector
image_vector = reshape(image, (768, 1))

# Reshape input data into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4, 3))

# Define x as a variable equal to 0.0
x = Variable(0.0)

# Define y as x*x and apply Gradient Tape
with GradientTape() as tape:
    tape.watch(x)
    y = multiply(x, x)
    
# Compute the gradient of y with respect to x
g = tape.gradient(y, x)

# Compute and print the gradient
print(g.numpy())

# Reshape b from a 1x3 to a 3x1 tensor
b = reshape(b, (3, 1))

# Multiply L by b
L1 = matmul(L, b)

# Sum over L1, evaluate, and print
L2 = reduce_sum(L1)
print(L2.numpy())

### 2. Linear Regression in TensorFlow

# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing 
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing.price)

# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing.waterfront, tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

# Import the keras module from tensorflow
from tensorflow import keras 

# Compute the mean squared error loss
loss = keras.losses.mse(price, predictions)

# Print the mean squared error
print(loss.numpy())

# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error loss
loss = keras.losses.mae(price, predictions)

# Print the mean squared error
print(loss.numpy())

# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define a loss function
def loss_function(scalar, features, target):
    # Define the predicted values
    predictions = scalar*features
    # Return the MAE loss
    return keras.losses.mae(target, predictions)

# Evaluate and print the loss function
print(loss_function(scalar, features, target).numpy())

# Define the intercept and slope
intercept = Variable(0.1, float32)
slope = Variable(0.1, float32)

# Set the loss function to take the variables as arguments
def loss_function(intercept, slope):
    # Set the predicted values
    pred_price_log = intercept+slope*lot_size_log
    # Return the MSE loss
    return keras.losses.mse(price_log, pred_price_log)

# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.1)

for j in range(1000):
    # Minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
    if j % 100 == 0:
        print(loss_function(intercept, slope).numpy())

# Print the intercept and slope
print(intercept.numpy(), slope.numpy())

# Define variables for intercept, slope_1, and slope_2
intercept = Variable(0.1, float32)
slope_1 = Variable(0.1, float32)
slope_2 = Variable(0.1, float32)

# Define the loss function
def loss_function(intercept, slope_1, slope_2):
    # Use the mean absolute error loss
    return keras.losses.mae(price_log, intercept+lot_size_log*slope_1+bedrooms*slope_2)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform one minimization step
opt.minimize(lambda: loss_function(intercept, slope_1, slope_2), var_list=[intercept, slope_1, slope_2])

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the loss function
def loss_function(intercept, slope, features, target):
    # Define the predicted values
    predictions = intercept+slope*features
    # Define the MSE loss  
    return keras.losses.mse(target, predictions)

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
    size_batch = np.array(batch['sqft_lot'], np.float32)
    # Extract the price values for the current batch
    price_batch = np.array(batch['price'], np.float32)
    # Minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope, size_batch, price_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())

