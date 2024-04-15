import carla
import time
import random

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

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Comment sentence below out if running the script for a second time
    world = client.load_world('Town04')

    try:
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        traffic_manager = client.get_trafficmanager()

        # Spawn two ambulances
        ambulance1 = setup_vehicle(world, 'vehicle.ford.ambulance', spawn_points[181], autopilot=False)
        ambulance2 = setup_vehicle(world, 'vehicle.ford.ambulance', spawn_points[172], autopilot=False, color='255,0,0')

        # Set spectator to focus on the AI ambulance
        if ambulance2:
            spectator = world.get_spectator()
            transform = ambulance2.get_transform()
            camera_transform = carla.Transform(transform.transform(carla.Location(x=-8, z=3)), transform.rotation)  # Adjust camera position as needed
            spectator.set_transform(camera_transform)

        # Define simple motion primitives (example: going straight for a brief period)
        if ambulance1:
            control = carla.VehicleControl(throttle=0.55, steer=0.0)
            ambulance1.apply_control(control)

        # Define simple motion primitives (example: going straight for a brief period)
        if ambulance2:
            control = carla.VehicleControl(throttle=0.7, steer=0.0)
            ambulance2.apply_control(control)

        # Run the scenario for a few seconds
        import time
        start_time = time.time()
        while time.time() - start_time < 300:
            world.wait_for_tick()  # assuming running in synchronous mode

    finally:
        # Clean up and reset the vehicles
        if ambulance1:
            ambulance1.destroy()
        if ambulance2:
            ambulance2.destroy()

        print("Scenario ended, cleaned up the vehicles.")

if __name__ == '__main__':
    main()
