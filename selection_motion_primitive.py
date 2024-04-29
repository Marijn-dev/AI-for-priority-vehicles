import numpy as np
import matplotlib.pyplot as plt

costmap = np.load(r'C:\Users\pepij\Documents\Master Year 1\Q3\5ARIP10 Interdisciplinary team project\costmap.npy')

# Generate a set of potential motion primitives
primitives = [
    {'curvature': 0, 'distance': 1}, # Go straight for 1 meters
    {'curvature': 0, 'distance': 2},
    {'curvature': 0, 'distance': 3},
    {'curvature': 5, 'distance': 1}, # turn right for 1 meter with 5 degrees per meter
    {'curvature': 5, 'distance': 2},
    {'curvature': 5, 'distance': 3},
    {'curvature': 10, 'distance': 1},
    {'curvature': 10, 'distance': 2},
    {'curvature': 10, 'distance': 3},
    {'curvature': 15, 'distance': 1},
    {'curvature': 15, 'distance': 2},
    {'curvature': 15, 'distance': 3},
    {'curvature': -5, 'distance': 1},
    {'curvature': -5, 'distance': 2},
    {'curvature': -5, 'distance': 3},
    {'curvature': -10, 'distance': 1},
    {'curvature': -10, 'distance': 2},
    {'curvature': -10, 'distance': 3},
    {'curvature': -15, 'distance': 1},
    {'curvature': -15, 'distance': 2},
    {'curvature': -15, 'distance': 3},
]

def calculate_primitive_costs(costmap, primitives, cell_size):
    # Function to calculate the cost of each motion primitive on the costmap
     return np.random.rand(len(primitives)) * 10  # Random cost between 0 and 10

def select_best_primitive(primitive_costs):
    # Function to select the primitive with the lowest cost
    min_cost_index = np.argmin(primitive_costs)
    return primitives[min_cost_index]

def plot_best_primitive(distance, curvature_deg_per_meter):
    # Check if the motion primitive is a straight line (curvature_deg_per_meter == 0)
    if curvature_deg_per_meter == 0:
        # Plot a straight line
        y = np.linspace(0, distance, num=300)
        x = np.zeros_like(y)
        plt.figure(figsize=(12, 12))
        plt.plot(x, y, 'b-', label=f'Straight Line: {distance:.2f} m')
    else:
        # Plot a curved path
        change_in_angle = curvature_deg_per_meter * distance  # Total angle change in degrees
        curvature_rad_per_meter = np.radians(curvature_deg_per_meter)  # Curvature in radians per meter
        radius = 1 / curvature_rad_per_meter  # Radius of the circle

        # Start and end angles of the arc
        start_angle = np.pi  # Start from the top
        end_angle = start_angle - np.radians(change_in_angle)  # End of the arc

        theta = np.linspace(start_angle, end_angle, num=300)
        x = radius * np.cos(theta) + radius  # Shift x by radius to start at (0,0)
        y = radius * np.sin(theta)

        plt.figure(figsize=(12, 12))
        plt.plot(x, y, 'b-', label=f'Arc Length: {distance:.2f} m, Angle: {change_in_angle} degrees')
    
    # Start and end points
    plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red')
    plt.text(x[0], y[0], "Start", fontsize=12, ha='center')
    plt.text(x[-1], y[-1], "End", fontsize=12, ha='center')

    # Labels and grid
    plt.title('Motion Primitive Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Show the plot
    plt.show()

def convert_to_vehicle_control(primitive):
    """
    Converts a motion primitive into VehicleControl settings for CARLA.
    
    Args:
    primitive (dict): Contains 'curvature' and 'distance' of the motion primitive.
    
    Returns:
    dict: Carla VehicleControl parameters including 'throttle', 'steer', and 'brake'.
    """
    # Placeholder function to convert curvature and distance to throttle and steer
    # These conversions would need to be calibrated based on vehicle behavior in CARLA
    throttle = 0.8  # Example fixed throttle for simplicity
    steer = np.clip(primitive['curvature'] / 17.5, -1, 1)  # Normalize and limit steer value
    brake = 0  # No braking in these examples
    
    return {'throttle': throttle, 'steer': steer, 'brake': brake}

def main(costmap, cell_size):
    # Calculate the costs of each primitive on the costmap
    primitive_costs = calculate_primitive_costs(costmap, primitives, cell_size)

    # Select the best primitive
    best_primitive = select_best_primitive(primitive_costs)

    # Plot the best primitive
    plot_best_primitive(best_primitive['distance'], best_primitive['curvature'])

    control = convert_to_vehicle_control(best_primitive)

    # Return the control values for the best primitive
    return control['throttle'], control['steer'], control['brake']

if __name__ == "__main__":
    cell_size = 0.1  # Size of the cells in meters (example value)
    throttle, steer, brake = main(costmap, cell_size)
    print(f"Throttle: {throttle}, Steer: {steer}, Brake: {brake}")
