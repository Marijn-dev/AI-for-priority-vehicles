import numpy as np
import matplotlib.pyplot as plt

costmap = np.load(r'C:\Users\pepij\Documents\Master Year 1\Q3\5ARIP10 Interdisciplinary team project\costmap.npy')
# rotated_costmap = np.flipud(costmap.T)

x_offset=0
y_offset=600
cell_size = 0.1
vehicle_width = 2.426
safety_margin = 0.2

# Generate a set of potential motion primitives
primitives = [
        {'curvature': 0, 'distance': 5},   # Go straight for 5 meters
        {'curvature': 0, 'distance': 7.5},
        {'curvature': 0, 'distance': 10},
        {'curvature': 5, 'distance': 5},
        {'curvature': 5, 'distance': 7.5},
        {'curvature': 5, 'distance': 10},
        {'curvature': 10, 'distance': 5},
        {'curvature': 10, 'distance': 7.5},
        {'curvature': 10, 'distance': 10},
        {'curvature': 15, 'distance': 5},
        {'curvature': 15, 'distance': 7.5},
        {'curvature': -5, 'distance': 5},
        {'curvature': -5, 'distance': 7.5},
        {'curvature': -5, 'distance': 10},
        {'curvature': -10, 'distance': 5},
        {'curvature': -10, 'distance': 7.5},
        {'curvature': -10, 'distance': 10},
        {'curvature': -15, 'distance': 5},
        {'curvature': -15, 'distance': 7.5},
    ]

def calculate_and_plot_best_primitive(costmap, primitives, cell_size, x_offset, y_offset, vehicle_width, safety_margin):
    costs = []
    for primitive in primitives:
        curvature_rad_per_meter = np.radians(primitive['curvature'])
        distance = primitive['distance']
        half_width = (vehicle_width / 2) + safety_margin  # Half width including safety margin

        if primitive['curvature'] != 0:
            radius = 1 / curvature_rad_per_meter
            change_in_angle = distance * curvature_rad_per_meter
            start_angle = -np.pi/2
            end_angle = start_angle - change_in_angle
            theta = np.linspace(start_angle, end_angle, num=300)

            x_center = (radius * (1 - np.cos(theta))) / cell_size + x_offset
            y_center = (radius * np.sin(theta)) / cell_size + y_offset

            x_start_adjustment = (radius * (1 - np.cos(start_angle))) / cell_size
            y_start_adjustment = (radius * (1 - np.sin(start_angle))) / cell_size
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

            x_indices = np.round(np.concatenate((x_left, x_right)) / cell_size).astype(int)
            y_indices = np.round(np.concatenate((y_left, y_right)) / cell_size).astype(int)

        else:
            x_center = np.linspace(0, distance / cell_size, num=300) + x_offset
            y_center = np.full_like(x_center, y_offset)

            x_indices = np.round(np.concatenate([x_center for _ in range(-int(half_width / cell_size), int(half_width / cell_size) + 1)])).astype(int)
            y_indices = np.round(np.concatenate([y_center + i for i in range(-int(half_width / cell_size), int(half_width / cell_size) + 1)])).astype(int)

        x_indices = np.clip(x_indices, 0, costmap.shape[1] - 1)
        y_indices = np.clip(y_indices, 0, costmap.shape[0] - 1)

        path_costs = costmap[y_indices, x_indices]
        average_cost = np.mean(path_costs) if len(path_costs) > 0 else np.inf
        normalized_cost = average_cost / distance  
        costs.append(normalized_cost)

    best_primitive_index = np.argmin(costs)
    best_primitive = primitives[best_primitive_index]

    if best_primitive['curvature'] == 0:
        y = np.linspace(0, best_primitive['distance'], num=300)
        x = np.zeros_like(y)
        plt.figure(figsize=(12, 12))
        plt.plot(x, y, 'b-', label=f'Straight Line: {best_primitive["distance"]:.2f} m')
    else:
        change_in_angle = best_primitive['curvature'] * best_primitive['distance']
        curvature_rad_per_meter = np.radians(best_primitive['curvature'])
        radius = 1 / curvature_rad_per_meter
        start_angle = np.pi
        end_angle = start_angle - np.radians(change_in_angle)
        theta = np.linspace(start_angle, end_angle, num=300)
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)
        plt.figure(figsize=(12, 12))
        plt.plot(x, y, 'b-', label=f'Arc Length: {best_primitive["distance"]:.2f} m, Angle: {change_in_angle} degrees')
    
    plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red')
    plt.text(x[0], y[0], "Start", fontsize=12, ha='center')
    plt.text(x[-1], y[-1], "End", fontsize=12, ha='center')
    plt.title('Motion Primitive Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return best_primitive

best_primitive = calculate_and_plot_best_primitive(costmap, primitives, cell_size, x_offset, y_offset, vehicle_width, safety_margin)
print("Best primitive:", best_primitive)
