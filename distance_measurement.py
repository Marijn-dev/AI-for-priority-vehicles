import carla
import time

# Function to spawn a vehicle
def setup_vehicle(world, model_id, spawn_point, autopilot=False, color=None):
    """Utility function to spawn a vehicle."""
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(model_id)
    if color:
        vehicle_bp.set_attribute('color', color)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        vehicle.set_autopilot(autopilot)
    return vehicle

def find_spawn_point(world):
    """
    Attempt to find a suitable spawn point on sidewalks or crosswalks.
    """
    spawn_location = carla.Location(x=202.4706573486328, y=-328.8112487792969, z=1.1225100755691528)
    spawn_rotation = carla.Rotation(pitch=0, yaw=88.68557739257812, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

# Function to apply motion primitive and measure distance
def measure_distance_for_primitive(world, vehicle, throttle, steer, acceleration_duration, measurement_duration):
    # Accelerate to top speed
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
    time.sleep(acceleration_duration)  # Allow the vehicle to reach top speed

    # Start measuring from this point
    start_location = vehicle.get_location()

    # Continue at top speed for the duration of the measurement
    time.sleep(measurement_duration)
    
    # Get the new location
    end_location = vehicle.get_location()
    
    # Measure the distance covered
    distance_covered = start_location.distance(end_location)
    
    # Reset vehicle control
    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1.0))
    time.sleep(1)  # Wait for the car to stop completely
    
    return distance_covered

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        ambulance_spawn_point = find_spawn_point(world)

        # Spawn the vehicle
        vehicle = setup_vehicle(world, 'vehicle.ford.ambulance', ambulance_spawn_point, autopilot=False, color='255,0,0')

        # Set spectator to focus on the AI ambulance
        if vehicle:
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            camera_transform = carla.Transform(transform.transform(carla.Location(x=-8, z=3)), transform.rotation)  # Adjust camera position as needed
            spectator.set_transform(camera_transform)
        
        if not vehicle:
            print("Could not spawn vehicle")
            return
        
        # List of motion primitives to measure (throttle, steer, acceleration duration, measurement duration)
        primitives = [
            (0.95, 0, 7, 5),  # Accelerate for 7 seconds, measure for 5 seconds
        ]
        
        # Measure distance for each primitive
        for throttle, steer, acceleration_duration, measurement_duration in primitives:
            distance = measure_distance_for_primitive(world, vehicle, throttle, steer, acceleration_duration, measurement_duration)
            print(f"Distance covered with throttle {throttle}, steer {steer} for {measurement_duration} seconds after {acceleration_duration} seconds of acceleration: {distance} meters")
        
    finally:
        # Clean up and destroy the vehicle
        if vehicle:
            vehicle.destroy()
        print("Scenario ended, cleaned up the vehicle.")

if __name__ == '__main__':
    main()
