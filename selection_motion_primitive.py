import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# Load the costmap
costmap = np.load(r'C:\Users\pepij\Documents\Master Year 1\Q3\5ARIP10 Interdisciplinary team project\costmap10.npy')
# rotated_costmap = np.flipud(costmap.T)

x_offset=0
y_offset=600

# Generate a set of potential motion primitives
primitives = [
        {'curvature': 0, 'distance': 10, 'velocity': 0.5},
        {'curvature': 0, 'distance': 10, 'velocity': 0.7},
        {'curvature': 0, 'distance': 10, 'velocity': 0.9},
        {'curvature': 0, 'distance': 20, 'velocity': 0.5},
        {'curvature': 0, 'distance': 20, 'velocity': 0.7},
        {'curvature': 0, 'distance': 20, 'velocity': 0.9},
        {'curvature': 2.5, 'distance': 10, 'velocity': 0.5},
        {'curvature': 2.5, 'distance': 20, 'velocity': 0.7},
        {'curvature': 5, 'distance': 10, 'velocity': 0.5},
        {'curvature': 5, 'distance': 20, 'velocity': 0.7},
        {'curvature': 10, 'distance': 10, 'velocity': 0.5},
        {'curvature': 10, 'distance': 10, 'velocity': 0.7},
        {'curvature': 15, 'distance': 10, 'velocity': 0.5},
        {'curvature': 15, 'distance': 10, 'velocity': 0.7},
        {'curvature': -2.5, 'distance': 10, 'velocity': 0.5},
        {'curvature': -2.5, 'distance': 20, 'velocity': 0.7},
        {'curvature': -5, 'distance': 10, 'velocity': 0.5},
        {'curvature': -5, 'distance': 20, 'velocity': 0.7},
        {'curvature': -10, 'distance': 10, 'velocity': 0.5},
        {'curvature': -10, 'distance': 10, 'velocity': 0.7},
        {'curvature': -15, 'distance': 10, 'velocity': 0.5},
        {'curvature': -15, 'distance': 10, 'velocity': 0.7},
    ]

def generate_random_colors(num_colors):
    np.random.seed(139)  # For reproducibility
    colors = np.random.rand(num_colors, 3)  # Generate random colors
    np.random.shuffle(colors)  # Shuffle to ensure randomness
    return colors

# def create_custom_colormap():
#     colors = [
#         'black',      # unlabeld things (0)
#         'gray',       # roads (1)
#         'lightcyan',        # sidewalks (2)
#         'lemonchiffon',     # buildings (3)
#         'skyblue',       # Wall (4)
#         'green',      # Fence (5)
#         'lightcoral',       # Pole (6)
#         'thistle',    # Traffic Lights (7)
#         'moccasin',     # Traffic Sign (8)
#         'tan',      # Vegetation (9)
#         'palegreen', # Terrain (10)
#         'skyblue',    # Sky (11)
#         'magenta',     # Pedestrian (12)
#         'pink',       # Rider (13)
#         'tomato',    # Cars (14)
#         'cornflowerblue',   # Trucks (15)
#         'darkgreen',  # Busses (16)
#         'gold',       # Train (17)
#         'lightgray',  # motorcycle (18)
#         'darkgray',   # Bicycle (19)
#         'lightblue',  # static objects (20)
#         'darkviolet', # Movable trash bins (21)
#         'lightyellow',# Other (22)
#         'navy',       # Water (23)
#         'lightpink',  # Roadlines (24)
#         'sandybrown',# Ground (25)
#         'teal',       # Bridge (26)
#         'olive',      # rail tracks (27)
#         'mistyrose'      # Guard rail (28)
#     ]
    
#     cmap = mcolors.ListedColormap(colors[:29])
#     return cmap

# def plot_costmap_with_labels(costmap):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     cmap = create_custom_colormap()
#     ax.imshow(costmap, cmap=cmap, interpolation='nearest')
#     plt.colorbar(ax.imshow(costmap, cmap=cmap, interpolation='nearest'), ax=ax)
#     ax.set_xlabel('x (cells)')
#     ax.set_ylabel('y (cells)')
#     ax.set_title('Costmap with Labels')
#     plt.show()

def plot_color_palette(colors):
    plt.figure(figsize=(10, 2))
    plt.imshow([colors], aspect='auto')
    plt.axis('off')
    plt.title('256 Randomly Shuffled Unique Colors')
    plt.show()

def plot_all_primitives(primitives):
    plt.figure(figsize=(12, 12))
    for primitive in primitives:
        distance = primitive['distance']
        curvature_deg_per_meter = primitive['curvature']

        if curvature_deg_per_meter == 0:
            # Straight line
            y = np.linspace(0, distance, num=300)
            x = np.zeros_like(y)
            plt.plot(x, y, label=f'Straight Line: {distance}m, Velocity: {primitive["velocity"]}m/s')
        else:
            # Curved path
            change_in_angle = curvature_deg_per_meter * distance
            curvature_rad_per_meter = np.radians(curvature_deg_per_meter)
            radius = 1 / curvature_rad_per_meter
            start_angle = np.pi  # Start from the top
            end_angle = start_angle - np.radians(change_in_angle)

            theta = np.linspace(start_angle, end_angle, num=300)
            x = radius * np.cos(theta) + radius
            y = radius * np.sin(theta)

            plt.plot(x, y, label=f'Arc: {distance}m, Angle: {change_in_angle}°, Velocity: {primitive["velocity"]}m/s')

        # Start and end points
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red')
        plt.text(x[0], y[0], "Start", fontsize=12, ha='center')
        plt.text(x[-1], y[-1], "End", fontsize=12, ha='center')

    # Labels and grid
    plt.title('All Motion Primitives')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    # plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def dynamic_safety_margin(velocity, base_margin=0.5, velocity_scale=0.8):
    return base_margin + velocity_scale * velocity
def time_penalty(velocity, base_penalty=10.0):
    return base_penalty / (1 + velocity)

def calculate_primitive_costs(costmap, primitives, cell_size, x_offset, y_offset, vehicle_width):
    costs = []
    for primitive in primitives:
        curvature_rad_per_meter = np.radians(primitive['curvature'])
        distance = primitive['distance']
        dynamic_margin = dynamic_safety_margin(primitive['velocity'])
        half_width = (vehicle_width / 2) + dynamic_margin
        penalty = time_penalty(primitive['velocity'])

        if primitive['curvature'] != 0:
            radius = 1 / curvature_rad_per_meter
            change_in_angle = distance * curvature_rad_per_meter
            start_angle = -np.pi / 2
            end_angle = start_angle - change_in_angle
            theta = np.linspace(start_angle, end_angle, num=300)

            x_center = (radius * (1 - np.cos(theta))) / cell_size + x_offset
            y_center = (radius * np.sin(theta)) / cell_size + y_offset

            x_start_adjustment = (radius * (1 - np.cos(start_angle))) / cell_size
            y_start_adjustment = (radius * (np.sin(start_angle))) / cell_size
            x_center -= x_start_adjustment
            y_center -= y_start_adjustment

            perpendicular_width = half_width / cell_size
            dx = -np.sin(theta) / cell_size
            dy = np.cos(theta) / cell_size

            norm_length = np.sqrt(dx**2 + dy**2)
            dx /= norm_length
            dy /= norm_length

            x_left = x_center + dy * perpendicular_width
            x_right = x_center - dy * perpendicular_width
            y_left = y_center - dx * perpendicular_width
            y_right = y_center + dx * perpendicular_width

            x_indices = np.round(np.concatenate((x_left, x_right))).astype(int)
            y_indices = np.round(np.concatenate((y_left, y_right))).astype(int)

        else:
            x_center = np.linspace(0, distance / cell_size, num=300) + x_offset
            y_center = np.full_like(x_center, y_offset)

            perpendicular_width = half_width / cell_size
            x_left = x_center
            x_right = x_center
            y_left = y_center + perpendicular_width
            y_right = y_center - perpendicular_width

            x_indices = np.round(np.concatenate((x_left, x_right))).astype(int)
            y_indices = np.round(np.concatenate((y_left, y_right))).astype(int)

        x_indices = np.clip(x_indices, 0, costmap.shape[1] - 1)
        y_indices = np.clip(y_indices, 0, costmap.shape[0] - 1)

        # Enhanced debugging output with pairs of coordinates
        print(f"\nPrimitive: {primitive}")

        # print("Center coordinates (x, y):")
        # for x, y in zip(x_center, y_center):
        #     print(f"({x}, {y})")

        # Check if indices are valid
        if len(x_indices) == 0 or len(y_indices) == 0:
            print(f"Invalid indices for primitive: {primitive}")
            costs.append(np.inf)
            continue

        # Check for index alignment
        if len(x_indices) != len(y_indices):
            print(f"Misaligned indices lengths for primitive: {primitive}")
            costs.append(np.inf)
            continue

        # print(f"Non-zero costmap cells: {np.count_nonzero(costmap)}")

        # Calculate path costs and print with indices
        path_costs = costmap[y_indices, x_indices]

        # print("Indices (x, y) and Path costs:")
        # index_cost_pairs = [(x, y, cost) for x, y, cost in zip(x_indices, y_indices, path_costs)]
        # print(index_cost_pairs)

        # print("Path costs with indices:")
        # for x, y, cost in zip(x_indices, y_indices, path_costs):
        #     print(f"Index (x, y): ({x}, {y}), Cost: {cost}")

        if len(path_costs) == 0:
            summed_cost = np.inf
        else:
            summed_cost = np.sum(path_costs)

        print(f"summed_cost: {summed_cost}")
        # print(f"Dynamic margin: {dynamic_margin}")
        # print(f"Penalty: {penalty}")

        normalized_cost = (summed_cost + dynamic_margin) / distance + penalty
        print(f"normalized_cost: {normalized_cost}")

        costs.append(normalized_cost)

    return np.array(costs)

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

def draw_motion_primitive_with_buffer(distance, curvature_deg_per_meter, vehicle_width, primitive):
    # Define figure
    plt.figure(figsize=(12, 12))

    # Calculate the dynamically adjusted safety margin
    dynamic_margin = dynamic_safety_margin(primitive['velocity'])

    # Setup for motion primitive
    if curvature_deg_per_meter == 0:  # Straight line case
        x = np.zeros(300)
        y = np.linspace(0, distance, num=300)
    else:  # Curved path
        change_in_angle = curvature_deg_per_meter * distance
        curvature_rad_per_meter = np.radians(curvature_deg_per_meter)
        radius = 1 / curvature_rad_per_meter
        start_angle = np.pi
        end_angle = start_angle - np.radians(change_in_angle)
        theta = np.linspace(start_angle, end_angle, num=300)
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)

    # Calculate normal directions for offsets
    dx = np.gradient(x)
    dy = np.gradient(y)
    norms = np.sqrt(dx**2 + dy**2)
    nx = -dy / norms
    ny = dx / norms

    # Calculate positions for the vehicle width and safety margins
    width_offset = vehicle_width / 2
    safety_offset = dynamic_margin
    x_vehicle_left = x + nx * width_offset
    y_vehicle_left = y + ny * width_offset
    x_vehicle_right = x - nx * width_offset
    y_vehicle_right = y - ny * width_offset
    x_safety_left = x + nx * (width_offset + safety_offset)
    y_safety_left = y + ny * (width_offset + safety_offset)
    x_safety_right = x - nx * (width_offset + safety_offset)
    y_safety_right = y - ny * (width_offset + safety_offset)

    # Plotting the main path
    plt.plot(x, y, 'b-', label='Center Path')
    plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red')  # Start and End markers
    plt.text(x[0], y[0], 'Start', fontsize=12, ha='center')
    plt.text(x[-1], y[-1], 'End', fontsize=12, ha='center')

    # Plotting the vehicle boundaries and safety margins
    plt.plot(x_vehicle_left, y_vehicle_left, 'r--', label='Left Vehicle Edge')
    plt.plot(x_vehicle_right, y_vehicle_right, 'r--', label='Right Vehicle Edge')
    plt.plot(x_safety_left, y_safety_left, 'g--', label='Left Safety Margin')
    plt.plot(x_safety_right, y_safety_right, 'g--', label='Right Safety Margin')

    # Setting up the plot
    plt.title('Motion Primitive with Vehicle Width and Safety Margins')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_best_primitive_costmap(ax, costmap, primitive, cell_size, x_offset, y_offset, vehicle_width):
    curvature_deg_per_meter = primitive['curvature']
    distance = primitive['distance']
    curvature_rad_per_meter = np.radians(curvature_deg_per_meter)
    # Calculate the dynamically adjusted safety margin
    dynamic_margin = dynamic_safety_margin(primitive['velocity'])
    half_width = (vehicle_width / 2) + dynamic_margin  # Half width including safety margin

    if curvature_deg_per_meter != 0:
        radius = 1 / curvature_rad_per_meter
        change_in_angle = distance * curvature_rad_per_meter
        start_angle = -np.pi/2
        end_angle = start_angle - change_in_angle
        theta = np.linspace(start_angle, end_angle, num=300)

        x_center = (radius * (1 - np.cos(theta))) / cell_size + x_offset  # adjust x to start from the x_offset
        y_center = (radius * np.sin(theta)) / cell_size + y_offset  # adjust y to start from the y_offset

        # Recalculate the starting position adjustments
        x_start_adjustment = (radius * (1 - np.cos(start_angle))) / cell_size
        y_start_adjustment = (radius * np.sin(start_angle)) / cell_size
        x_center -= x_start_adjustment
        y_center -= y_start_adjustment

        # Calculate derivatives
        dx = np.sin(theta) / cell_size  # x derivative
        dy = np.cos(theta) / cell_size   # y derivative

        # Calculate normal vectors for left and right sides
        norm_length = np.sqrt(dx**2 + dy**2)
        dx /= norm_length
        dy /= norm_length

        # Calculate offsets for buffers and safety margins
        perpendicular_width = half_width / cell_size
        x_buffer_left = x_center + dy * perpendicular_width
        y_buffer_left = y_center - dx * perpendicular_width
        x_buffer_right = x_center - dy * perpendicular_width
        y_buffer_right = y_center + dx * perpendicular_width

        x_safety_left = x_buffer_left + dy * (dynamic_margin / cell_size)
        y_safety_left = y_buffer_left - dx * (dynamic_margin / cell_size)
        x_safety_right = x_buffer_right - dy * (dynamic_margin / cell_size)
        y_safety_right = y_buffer_right + dx * (dynamic_margin / cell_size)

        # Plotting all lines
        ax.plot(x_center, y_center, 'b-', label='Motion Primitive')
        ax.plot(x_buffer_left, y_buffer_left, 'r--', label='Left Buffer')
        ax.plot(x_buffer_right, y_buffer_right, 'r--', label='Right Buffer')
        ax.plot(x_safety_left, y_safety_left, 'g--', label='Left Safety Margin')
        ax.plot(x_safety_right, y_safety_right, 'g--', label='Right Safety Margin')
    else:
        # Straight motion primitive setup
        x_center = np.linspace(0, distance / cell_size, num=300) + x_offset # Convert to grid scale
        y_center = np.full_like(x_center, y_offset)  # Only y-offset

        # Offset calculation for straight paths
        perpendicular_width = half_width / cell_size
        safety_margin_cells = dynamic_margin / cell_size

        # Since the motion is straight along the x-axis, only y-offsets are needed
        y_buffer_left = y_center + perpendicular_width
        y_buffer_right = y_center - perpendicular_width
        y_safety_left = y_center + perpendicular_width + safety_margin_cells
        y_safety_right = y_center - perpendicular_width - safety_margin_cells

        # Plotting all lines
        ax.plot(x_center, y_center, 'b-', label='Motion Primitive')
        ax.plot(x_center, y_buffer_left, 'r--', label='Left Buffer')
        ax.plot(x_center, y_buffer_right, 'r--', label='Right Buffer')
        ax.plot(x_center, y_safety_left, 'g--', label='Left Safety Margin')
        ax.plot(x_center, y_safety_right, 'g--', label='Right Safety Margin')

    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    ax.legend()

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
    throttle = primitive['velocity']
    steer = np.clip(primitive['curvature'] / 17.5, -1, 1)  # Normalize and limit steer value
    brake = 0  # No braking in these examples
    
    return {'throttle': throttle, 'steer': steer, 'brake': brake}

def main(costmap, cell_size):
    num_colors = 256
    random_colors = generate_random_colors(num_colors)
    # plot_color_palette(random_colors)  
    cmap = LinearSegmentedColormap.from_list("random_cmap", random_colors, N=256)
    # cmap = create_custom_colormap()

    # plot_all_primitives(primitives)

    # Calculate the costs of each primitive on the costmap
    primitive_costs = calculate_primitive_costs(costmap, primitives, cell_size=0.1, x_offset=0, y_offset=600, vehicle_width=2.426)
    print("primitive_costs are:"+str(primitive_costs))

    # Select the best primitive
    best_primitive = select_best_primitive(primitive_costs)
    # best_primitive = primitives[0]
    print("best primitive is:"+str(best_primitive))

    # # Plot the best primitive
    # plot_best_primitive(best_primitive['distance'], best_primitive['curvature'])

    draw_motion_primitive_with_buffer(best_primitive['distance'], best_primitive['curvature'], 2.426, best_primitive)

    # plot_costmap_with_labels(costmap)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(costmap, cmap=cmap, interpolation='nearest')
    plot_best_primitive_costmap(ax, costmap, best_primitive, cell_size=0.1, x_offset=0 , y_offset=600, vehicle_width=2.426)
    # Set limits for the axes
    ax.set_xlim([0, 1000])
    ax.set_ylim([800, 400])
    plt.show()

    control = convert_to_vehicle_control(best_primitive)

    # Return the control values for the best primitive
    return control['throttle'], control['steer'], control['brake']

if __name__ == "__main__":
    cell_size = 0.1  # Size of the cells in meters (example value)
    throttle, steer, brake = main(costmap, cell_size)
    print(f"Throttle: {throttle}, Steer: {steer}, Brake: {brake}")
