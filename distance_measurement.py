import carla
import time

# Function to spawn a vehicle
def spawn_vehicle(world, vehicle_type='model3', color='255,0,0'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.ford.ambulance')
    vehicle_bp.set_attribute('color', color)
    spawn_points = world.get_map().get_spawn_points()
    return world.try_spawn_actor(vehicle_bp, spawn_points[0])

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
        # Spawn the vehicle
        vehicle = spawn_vehicle(world)

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
            (0.8, 0, 7, 5),  # Accelerate for 7 seconds, measure for 5 seconds
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
