import numpy as np

def create_motion_primitives():
    """
    Generates a set of motion primitives based on specified curvatures and distances.
    
    Returns:
    list of dicts: Each dictionary contains 'curvature' and 'distance' for a motion primitive.
    """
    # Example primitives, curvature in degrees per meter, distance in meters
    primitives = [
        {'curvature': 0, 'distance': 5},   # Go straight for 10 meters
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
    return primitives

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

def main():
    # Create motion primitives
    primitives = create_motion_primitives()
    
    # Convert each primitive to vehicle control commands
    for primitive in primitives:
        control = convert_to_vehicle_control(primitive)
        print(f"Primitive: {primitive}, Control: {control}")

if __name__ == "__main__":
    main()
