import carla
import time

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
    """Attempt to find a suitable spawn point."""
    spawn_location = carla.Location(x=202.4706573486328, y=-318.8112487792969, z=1.1225100755691528)
    spawn_rotation = carla.Rotation(pitch=0, yaw=91, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def measure_distance_and_yaw_for_primitive(world, vehicle, throttle, steer, desired_distance):
    """Function to apply motion primitive and measure distance and change in yaw."""
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
    start_location = vehicle.get_location()
    initial_yaw = vehicle.get_transform().rotation.yaw

    distance_covered = 0.0
    while distance_covered < desired_distance:
        world.wait_for_tick()  # Assumes synchronous mode
        current_location = vehicle.get_location()
        distance_covered = start_location.distance(current_location)

    final_yaw = vehicle.get_transform().rotation.yaw
    yaw_change = final_yaw - initial_yaw

    # Adjust for yaw wrapping around 360 degrees
    if yaw_change > 180:
        yaw_change -= 360
    elif yaw_change < -180:
        yaw_change += 360

    # Stop the vehicle
    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1.0))
    return distance_covered, yaw_change

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        vehicle_spawn_point = find_spawn_point(world)
        vehicle = setup_vehicle(world, 'vehicle.ford.ambulance', vehicle_spawn_point, color='255,0,0')

        if not vehicle:
            print("Could not spawn vehicle")
            return

        # Set spectator to focus on the vehicle
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        camera_transform = carla.Transform(transform.transform(carla.Location(x=-8, z=3)), transform.rotation)  # Adjust camera position as needed
        spectator.set_transform(camera_transform)

        # Define motion primitive parameters
        throttle = 0.8
        steer = 0.8571428571428571
        desired_distance = 1  # Desired distance in meters

        # Measure distance and yaw change
        distance, yaw_change = measure_distance_and_yaw_for_primitive(world, vehicle, throttle, steer, desired_distance)
        print(f"Distance covered: {distance:.2f} meters")
        print(f"Change in yaw: {yaw_change:.2f} degrees")

    finally:
        if vehicle:
            vehicle.destroy()
        print("Scenario ended, vehicle cleaned up.")

if __name__ == '__main__':
    main()
