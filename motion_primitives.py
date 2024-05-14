import carla
import random
import time

def set_synchronous_mode(world, synchronous_mode, delta_seconds=0.025):
    """
    Set the CARLA world to run in synchronous or asynchronous mode.
    """
    settings = world.get_settings()
    settings.synchronous_mode = synchronous_mode
    settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(settings)

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

# Enable synchronous mode (set to False to disable)
synchronous_mode = True
set_synchronous_mode(world, synchronous_mode)

# Get a blueprint for the vehicle
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points() 

# Get the blueprint for the vehicle you want
vehicle_bp = bp_lib.find('vehicle.ford.ambulance') 

# Try spawning the vehicle at a randomly chosen spawn point
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Move the spectator behind the vehicle 
spectator = world.get_spectator() 
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
spectator.set_transform(transform)

# Define a simple lookup table for motion primitives
motion_primitives = {
    # (throttle, steer, brake)
    'accelerate_gentle': (0.5, 0.0, 0.0),
    'accelerate_hard': (1.0, 0.0, 0.0),
    'brake_gentle': (0.0, 0.0, 0.5),
    'brake_hard': (0.0, 0.0, 1.0),
    'reverse_15': (-0.15, 0.0, 0.0),
    'go_straight_15': (0.15, 0.0, 0.0),
    'go_straight_30': (0.30, 0.0, 0.0),
    'go_straight_45': (0.45, 0.0, 0.0),
    'go_straight_60': (0.60, 0.0, 0.0),
    'turn_right_15_small': (0.15, 0.3, 0.0),
    'turn_right_15_medium': (0.15, 0.5, 0.0),
    'turn_right_15_large': (0.15, 0.7, 0.0),
    'turn_right_30_small': (0.30, 0.3, 0.0),
    'turn_right_30_medium': (0.30, 0.5, 0.0),
    'turn_right_30_large': (0.30, 0.7, 0.0),
    'turn_right_45_small': (0.45, 0.3, 0.0),
    'turn_right_45_medium': (0.45, 0.5, 0.0),
    'turn_right_45_large': (0.45, 0.7, 0.0),
    'turn_left_15_small': (0.15, -0.3, 0.0),
    'turn_left_15_medium': (0.15, -0.5, 0.0),
    'turn_left_15_large': (0.15, -0.7, 0.0),
    'turn_left_30_small': (0.30, -0.3, 0.0),
    'turn_left_30_medium': (0.30, -0.5, 0.0),
    'turn_left_30_large': (0.30, -0.7, 0.0),
    'turn_left_45_small': (0.45, -0.3, 0.0),
    'turn_left_45_medium': (0.45, -0.5, 0.0),
    'turn_left_45_large': (0.45, -0.7, 0.0),
}

# Function to apply a motion primitive
def apply_motion_primitive_sync(vehicle, primitive_name, duration_seconds, world):
    """
    Apply the specified motion primitive to the vehicle for a given duration in seconds.
    In synchronous mode, the world is ticked forward for each step of the duration.
    """
    throttle, steer, brake = motion_primitives[primitive_name]
    control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    
    # Calculate the number of ticks based on the duration and delta_seconds
    ticks = int(duration_seconds / world.get_settings().fixed_delta_seconds)

    # spectator = world.get_spectator()
    
    for _ in range(ticks):
        vehicle.apply_control(control)
        world.tick()

        # Update the spectator's position to follow the vehicle
        transform = vehicle.get_transform()
        follow_distance = carla.Location(x=-4, z=2.5)  # Adjust as needed to position the camera
        spectator_transform = carla.Transform(transform.location + follow_distance, transform.rotation)
        spectator.set_transform(spectator_transform)

def apply_motion_primitive_async(vehicle, primitive_name, duration_seconds):
    start_time = time.time()
    throttle, steer, brake = motion_primitives[primitive_name]
    control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    while time.time() - start_time < duration_seconds:
        vehicle.apply_control(control)
        # time.sleep(0.00005)


# Use the function to apply a motion primitive
apply_motion_primitive_sync(vehicle, 'turn_left_45_small', duration_seconds=100, world=world)
# apply_motion_primitive_async(vehicle, 'go_straight_45', duration_seconds=100, world=world)

# Clean up: remember to destroy the vehicle actor once done
# vehicle.destroy()

# Reset to asynchronous mode after the script is done
set_synchronous_mode(world, False)