import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Sample data for a simple turn
steering_angle = np.array([0.4, 0.4, 0.4])  # Steering angle
throttle = np.array([0.8, 0.8, 0.8])  # Throttle value
durations = np.array([3, 4, 5])  # Duration of the turn (seconds)
curvatures = np.array([23.387977600097656, 48.268394470214844, 88.61212921142578])  # Measured curvatures for the corresponding durations

# Combine your input data into a single feature matrix
X = durations.reshape(-1, 1)  # Reshape to make it a column vector

# Create a model pipeline with polynomial features and linear regression
degree = 2  # You can adjust the degree of the polynomial based on your data
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the model to your data
model.fit(X, curvatures)

# Now you can predict curvature based on duration using the model
# For example, predicting for a duration of 6 seconds:
duration = 6
predicted_curvature = model.predict(np.array([[duration]]))
print(f"Predicted curvature for a duration of {duration} seconds: {predicted_curvature[0]}")
