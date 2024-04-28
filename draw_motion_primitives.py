import numpy as np
import matplotlib.pyplot as plt

# Constants for the motion primitive
distance = 5  # in meters
curvature_deg_per_meter = 5  # in degrees per meter

# Calculate the change in angle over the given distance
change_in_angle = curvature_deg_per_meter * distance  # in degrees

# Calculate the radius of the circle that this arc is part of
# Formula: radius = 1 / curvature, but first convert curvature to radians per meter
curvature_rad_per_meter = np.radians(curvature_deg_per_meter)
radius = 1 / curvature_rad_per_meter

# Calculate the arc length
arc_length = radius * np.radians(change_in_angle)

# The start and end angle of the arc on the circle
start_angle = np.pi  # start from the top of the circle
end_angle = start_angle - np.radians(change_in_angle)

# Parametric equations for the circle segment (arc)
theta = np.linspace(start_angle, end_angle, num=300)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

x = x + radius

# Plot the motion primitive
plt.figure(figsize=(12, 12))
plt.plot(x, y, label=f'Arc Length: {arc_length:.2f} m, Angle: {change_in_angle} degrees')
plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red')  # Start and end points

# Annotations and labels
plt.title('Motion Primitive Trajectory')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure the aspect ratio is square

# Show the plot
plt.show()
