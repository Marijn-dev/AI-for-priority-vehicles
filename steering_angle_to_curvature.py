import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Sample data for a simple turn
steering_angles = np.array([0.4, 0.4, 0.4])  # Steering angles
throttles = np.array([0.8, 0.8, 0.8])  # Throttle values
durations = np.array([3, 4, 5])  # Duration of the turns (seconds)
curvatures = np.array([23.387977600097656, 48.268394470214844, 88.61212921142578])  # Measured curvatures

# Combine input data into a single feature matrix
X = np.vstack((steering_angles, throttles, durations)).T

# Create a model pipeline
degree = 2  # Polynomial degree
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the model
model.fit(X, curvatures)

def predict_curvature(steering_angle, throttle, duration):
    """
    Predict the curvature based on steering angle, throttle, and duration.

    Args:
    steering_angle (float): The steering angle value.
    throttle (float): The throttle value.
    duration (float): The duration of the turn.

    Returns:
    float: Predicted curvature.
    """
    # Prepare the feature array from inputs
    features = np.array([[steering_angle, throttle, duration]])
    
    # Predict using the trained model
    predicted_curvature = model.predict(features)[0]
    
    return predicted_curvature

# Example of how to use the predict function
if __name__ == "__main__":
    example_curvature = predict_curvature(0.4, 0.8, 4)
    print(f"Predicted curvature: {example_curvature}")
