# Importing necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# Loading the dataset
dataset = pd.read_csv('forestfires.csv')

# Displaying the first few rows of the dataset
print(dataset.head())

# Checking for categorical columns
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)

# Encoding categorical variables
ordinal_encoder = OrdinalEncoder()
dataset[object_cols] = ordinal_encoder.fit_transform(dataset[object_cols])

# Displaying the dataset after encoding
print(dataset.head())

# Separating features (x) and target (y)
x = dataset.drop(columns = ['area'])
y = dataset['area']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Building the neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape = (x_train.shape[1],), activation='relu' ))
model.add(tf.keras.layers.Dense(256, activation='relu' ))
model.add(tf.keras.layers.Dense(1, activation='tanh' ))

# Compiling the model
model.compile(optimizer= 'adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, epochs=5000)

# Predict on test data
y_pred = model.predict(x_test)

# Calculating metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Printing the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r_squared}")
print(f"Median Absolute Error (MedAE): {medae}")
