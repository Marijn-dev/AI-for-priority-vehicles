import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# Load the costmap
# costmap = np.load(r'C:\Users\20192651\Documents\Master Year 1\Q3\5ARIP10 Interdisciplinary team project\costmap11.npy')
# rotated_costmap = np.flipud(costmap.T)

x_offset=0
y_offset=600

target_coordinates = {
        '1': (200, -278), # town
        '2': (235, -307), # side town
        '3': (166, -311), # houses
        '4': (205, -341) # highway
    }

target_direction = '1'  # Set the desired direction here
target = target_coordinates[target_direction]

# Generate a set of potential motion primitives
primitives = [
        {'curvature': 0, 'distance': 10, 'velocity': 0.5},
        {'curvature': 0, 'distance': 10, 'velocity': 0.65},
        {'curvature': 0, 'distance': 20, 'velocity': 0.75},
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

def dynamic_safety_margin(velocity, base_margin=0.0, velocity_scale=0.8):
    return base_margin + velocity_scale * velocity

def time_penalty(velocity, base_penalty=10.0):
    return base_penalty / (velocity)

def calculate_final_global_coordinates(ambulance_location, ambulance_rotation, distance, curvature_rad_per_meter):
    if curvature_rad_per_meter == 0:
        # Straight line
        final_x = ambulance_location.x + distance * np.cos(np.radians(ambulance_rotation.yaw))
        final_y = ambulance_location.y + distance * np.sin(np.radians(ambulance_rotation.yaw))
    else:
        # Curved path
        radius = 1 / curvature_rad_per_meter
        change_in_angle = distance * curvature_rad_per_meter
        angle = np.radians(ambulance_rotation.yaw) + change_in_angle

        final_x = ambulance_location.x + radius * (np.sin(angle) - np.sin(np.radians(ambulance_rotation.yaw)))
        final_y = ambulance_location.y - radius * (np.cos(angle) - np.cos(np.radians(ambulance_rotation.yaw)))

    return final_x, final_y

def calculate_gps_cost(final_x, final_y, target_x, target_y):
    distance = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
    return distance

def create_collision_map(participants_labels, participants_positions, prediction_horizon, cost_map):
    M = [np.zeros_like(cost_map) for _ in range(prediction_horizon)]
    for t in range(prediction_horizon):
        for p in range(len(participants_labels)):
            place_traffic_participants(t, participants_labels[p], participants_positions[t][p], M)
    return M

def place_traffic_participants(t, label, position, M):
    x_car, y_car = position

    actor_radius = 10 if label == 'car' else 5

    for x in range(max(0, x_car - actor_radius), min(M[t].shape[0], x_car + actor_radius)):
        for y in range(max(0, y_car - actor_radius), min(M[t].shape[1], y_car + actor_radius)):
            M[t][x, y] = 512

def calculate_primitive_costs(costmap, predicted_costmaps, primitives, cell_size, x_offset, y_offset, vehicle_width, ambulance_location, ambulance_rotation, target):
    costs = []
    for primitive in primitives:
        curvature_rad_per_meter = np.radians(primitive['curvature'])
        distance = primitive['distance']
        dynamic_margin = dynamic_safety_margin(primitive['velocity'])
        half_width = ((vehicle_width + dynamic_margin) / 2)
        penalty = time_penalty(primitive['velocity'])
        target_x, target_y = target

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

        else:
            x_center = np.linspace(0, distance / cell_size, num=300) + x_offset
            y_center = np.full_like(x_center, y_offset)

            perpendicular_width = half_width / cell_size
            x_left = x_center
            x_right = x_center
            y_left = y_center + perpendicular_width
            y_right = y_center - perpendicular_width

        x_indices = np.clip(np.round(np.concatenate((x_left, x_right))).astype(int), 0, costmap.shape[1] - 1)
        y_indices = np.clip(np.round(np.concatenate((y_left, y_right))).astype(int), 0, costmap.shape[0] - 1)

        # Include all cells within the safety margins
        all_indices = set()
        for x_start, y_start, x_end, y_end in zip(x_left, y_left, x_right, y_right):
            num_points = max(int(np.hypot(x_end - x_start, y_end - y_start)), 100)
            x_line = np.linspace(x_start, x_end, num=num_points).astype(int)
            y_line = np.linspace(y_start, y_end, num=num_points).astype(int)
            for x, y in zip(x_line, y_line):
                all_indices.update([(xi, yi) for xi in range(x - int(perpendicular_width), x + int(perpendicular_width) + 1) for yi in range(y - int(perpendicular_width), y + int(perpendicular_width) + 1)])

        x_all_indices, y_all_indices = zip(*all_indices)
        x_all_indices = np.clip(x_all_indices, 0, costmap.shape[1] - 1)
        y_all_indices = np.clip(y_all_indices, 0, costmap.shape[0] - 1)

        # Calculate path costs
        path_costs = costmap[y_all_indices, x_all_indices]

        if len(path_costs) == 0:
            summed_cost = np.inf
        else:
            summed_cost = np.sum(path_costs)

        normalized_cost = summed_cost / distance + penalty

        # Calculate the final global coordinates of the primitive
        x_final_global, y_final_global = calculate_final_global_coordinates(ambulance_location, ambulance_rotation, distance, curvature_rad_per_meter)
        gps_cost = calculate_gps_cost(x_final_global, y_final_global, target_x, target_y)

        collision_cost = 0
        for t in range(len(predicted_costmaps)):
            segment_length = len(x_all_indices) // len(predicted_costmaps)
            segment_x_indices = x_all_indices[t * segment_length:(t + 1) * segment_length]
            segment_y_indices = y_all_indices[t * segment_length:(t + 1) * segment_length]

            collision_cost += np.sum(predicted_costmaps[t][segment_y_indices, segment_x_indices])

        total_cost = normalized_cost #+ gps_cost + collision_cost

        costs.append(total_cost)

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
    half_width = (vehicle_width + dynamic_margin) / 2  # Half width including safety margin

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

# Define the function to plot segments of the best primitive
def plot_primitive_segment_on_prediction_costmaps(costmap, predicted_costmaps, primitive, cell_size, x_offset, y_offset, vehicle_width, segment_index):
    curvature_deg_per_meter = primitive['curvature']
    distance = primitive['distance']
    curvature_rad_per_meter = np.radians(curvature_deg_per_meter)
    dynamic_margin = dynamic_safety_margin(primitive['velocity'])
    half_width = (vehicle_width / 2) + dynamic_margin  # Half width including safety margin

    if curvature_deg_per_meter != 0:
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
        dx = np.sin(theta) / cell_size
        dy = np.cos(theta) / cell_size

        norm_length = np.sqrt(dx**2 + dy**2)
        dx /= norm_length
        dy /= norm_length

        x_left = x_center + dy * perpendicular_width
        x_right = x_center - dy * perpendicular_width
        y_left = y_center - dx * perpendicular_width
        y_right = y_center + dx * perpendicular_width

        x_safety_left = x_left + dy * (dynamic_margin / cell_size)
        y_safety_left = y_left - dx * (dynamic_margin / cell_size)
        x_safety_right = x_right - dy * (dynamic_margin / cell_size)
        y_safety_right = y_right + dx * (dynamic_margin / cell_size)

    else:
        x_center = np.linspace(0, distance / cell_size, num=300) + x_offset
        y_center = np.full_like(x_center, y_offset)

        perpendicular_width = half_width / cell_size
        safety_margin_cells = dynamic_margin / cell_size

        x_left = x_center
        x_right = x_center
        y_left = y_center + perpendicular_width
        y_right = y_center - perpendicular_width

        x_safety_left = y_left + safety_margin_cells
        y_safety_left = y_left + safety_margin_cells
        x_safety_right = y_right - safety_margin_cells
        y_safety_right = y_right - safety_margin_cells

    # Segment indices
    segment_length = 300 // len(predicted_costmaps)
    start_idx = segment_index * segment_length
    end_idx = start_idx + segment_length

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(predicted_costmaps[segment_index], cmap=cmap, interpolation='nearest')
    ax.plot(x_center[start_idx:end_idx], y_center[start_idx:end_idx], 'b-', label='Motion Primitive')
    ax.plot(x_left[start_idx:end_idx], y_left[start_idx:end_idx], 'r--', label='Left Buffer')
    ax.plot(x_right[start_idx:end_idx], y_right[start_idx:end_idx], 'r--', label='Right Buffer')
    ax.plot(x_safety_left[start_idx:end_idx], y_safety_left[start_idx:end_idx], 'g--', label='Left Safety Margin')
    ax.plot(x_safety_right[start_idx:end_idx], y_safety_right[start_idx:end_idx], 'g--', label='Right Safety Margin')
    ax.set_xlim([0, 1000])
    ax.set_ylim([800, 400])
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    ax.legend()
    ax.set_title(f'Primitive Segment on Predicted Costmap {segment_index}')
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
    throttle = primitive['velocity']
    steer = np.clip(primitive['curvature'] / 17.5, -1, 1)  # Normalize and limit steer value
    brake = 0  # No braking in these examples
    
    return throttle,steer,brake
    #return {'throttle': throttle, 'steer': steer, 'brake': brake}

def main(costmap, predicted_costmaps, cell_size, target):
    num_colors = 256
    random_colors = generate_random_colors(num_colors)
    # plot_color_palette(random_colors)  
    cmap = LinearSegmentedColormap.from_list("random_cmap", random_colors, N=256)
    # cmap = create_custom_colormap()

    # plot_all_primitives(primitives)

    # Calculate the costs of each primitive on the costmap
    # primitive_costs = calculate_primitive_costs(costmap, predicted_costmaps, primitives, cell_size=0.1, x_offset=0, y_offset=600, vehicle_width=2.426, target=target)
    # print("primitive_costs are:"+str(primitive_costs))

    # Select the best primitive
    # best_primitive = select_best_primitive(primitive_costs)
    # best_primitive = primitives[0]
    # print("best primitive is:"+str(best_primitive))

    # # Plot the best primitive
    # plot_best_primitive(best_primitive['distance'], best_primitive['curvature'])

    # draw_motion_primitive_with_buffer(best_primitive['distance'], best_primitive['curvature'], 2.426, best_primitive)

    # plot_costmap_with_labels(costmap)

    # create_collision_map(participants_labels=['car', 'pedestrian'],participants_positions=participants_positions,time_step=1,ego_position=(0,0),cost_map=costmap,prediction_horizon=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(costmap, cmap=cmap, interpolation='nearest')
    plot_best_primitive_costmap(ax, costmap, best_primitive, cell_size, x_offset=0 , y_offset=600, vehicle_width=2.426)
    # Set limits for the axes
    ax.set_xlim([0, 1000])
    ax.set_ylim([800, 400])
    plt.show()

    throttle, steer, brake = convert_to_vehicle_control(best_primitive)

    return throttle, steer, brake

if __name__ == "__main__":
    cell_size = 0.1  # Size of the cells in meters (example value)# Set the target coordinates for 'straight', 'left', and 'right'
    num_colors = 256
    random_colors = generate_random_colors(num_colors)
    cmap = LinearSegmentedColormap.from_list("random_cmap", random_colors, N=256)
    prediction_horizon = 5
    participants_labels = ['car1', 'car2']
    participants_positions = [  # Positions at each timestep, x and y are swapped for some reason
        [[195, 547], [258, 628]],
        [[172, 551], [258, 609]],
        [[149, 555], [258, 590]],
        [[126, 559], [258, 571]],
        [[103, 563], [258, 552]]
    ]
    # Swap the x and y values
    swapped_positions = [[[pos[1], pos[0]] for pos in timestep] for timestep in participants_positions]

    predicted_costmaps = create_collision_map(participants_labels, swapped_positions, prediction_horizon, costmap)
    # # Print and visualize predicted costmaps for debugging
    # for t, predicted_costmap in enumerate(predicted_costmaps):
    #     print(f"Predicted costmap at timestep {t}:")
    #     print(predicted_costmap)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(predicted_costmap, cmap=cmap, interpolation='nearest')
    #     plt.xlim(0,1000)
    #     plt.ylim(800,400)
    #     plt.title(f"Predicted costmap at timestep {t}")
    #     plt.colorbar()
    #     plt.show()

    primitive_costs = calculate_primitive_costs(costmap, predicted_costmaps, primitives, cell_size, x_offset, y_offset, 2.426, target)
    print("primitive_costs are:"+str(primitive_costs))
    best_primitive = select_best_primitive(primitive_costs)
    print("best primitive is:", best_primitive)

    # # Plot the segments
    # for segment_index in range(prediction_horizon):
    #     plot_primitive_segment_on_prediction_costmaps(costmap, predicted_costmaps, best_primitive, cell_size, x_offset, y_offset, 2.426, segment_index)

    throttle, steer, brake = main(costmap, predicted_costmaps, cell_size, target)
    print(f"Throttle: {throttle}, Steer: {steer}, Brake: {brake}")
