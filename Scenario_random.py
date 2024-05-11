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
    return vehicle, autopilot

def setup_pedestrian(world, spawn_point):
    """Utility function to spawn a pedestrian."""
    blueprint_library = world.get_blueprint_library()
    pedestrian_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
    return pedestrian

def find_spawn_point_1(world):
    spawn_location = carla.Location(x=204.56409912109375, y=-278.99392700195312 + random.randrange(-20, 20,2), z=0.7819424271583557)
    spawn_rotation = carla.Rotation(pitch=0, yaw=-88.68557739257812, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_spawn_point_2(world):
    spawn_location = carla.Location(x=202.4706573486328, y=-335.8112487792969 + random.randrange(-20, 20,2), z=1.1225100755691528)
    spawn_rotation = carla.Rotation(pitch=0, yaw=91, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_spawn_point_3(world):
    spawn_location = carla.Location(x=224.24923706054688+random.randrange(-20, 20,2), y=-310.7073059082031, z=1.3423326015472412)
    spawn_rotation = carla.Rotation(pitch=0, yaw=178.68557739257812, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_spawn_point_4(world):
    spawn_location = carla.Location(x=177.24923706054688+random.randrange(-20, 20,2), y=-307.7073059082031, z=1.3423326015472412)
    spawn_rotation = carla.Rotation(pitch=-6, yaw=-1.1, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_pedestrian_spawn_point(world):
    """
    Attempt to find a suitable pedestrian spawn point on sidewalks or crosswalks.
    """
    spawn_location = carla.Location(x=196, y=-311, z=5)
    spawn_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Comment sentence below out if running the script for a secwond time
    # world = client.load_world('Town04')w
    
    # clean up actors if any are left
    actors = world.get_actors()

    # Iterate through actors
    for actor in actors:
        # Check if actor is a vehicle
        if actor.type_id.startswith('vehicle'):
            # Destroy the vehicle
            actor.destroy()
    try:
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        spawn_point_pool = [find_spawn_point_1(world), find_spawn_point_2(world), find_spawn_point_3(world), find_spawn_point_4(world)]

        # Shuffle the pool to randomize order
        # random.shuffle(spawn_point_pool)

        traffic_manager = client.get_trafficmanager()
    
        # Randomly choose spawn points for each participant
        ai_ambulance_spawn_point = spawn_point_pool.pop()
        ambulance_spawn_point = spawn_point_pool.pop()
        car_spawn_point_1 = spawn_point_pool.pop()
        car_spawn_point_2 = spawn_point_pool.pop()

        pedestrian_spawn_point = find_pedestrian_spawn_point(world)

        # Use line below to get coordinates of a spawn point
        # print(f"Location: {spawn_points[172].location.x}, {spawn_points[172].location.y}, {spawn_points[172].location.z}, Rotation: {spawn_points[172].rotation.pitch}, {spawn_points[172].rotation.yaw}, {spawn_points[172].rotation.roll}")

        # Spawn two ambulances
        ai_ambulance, ai_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ai_ambulance_spawn_point, autopilot=True, color='255,0,0')
        human_ambulance, human_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ambulance_spawn_point, autopilot=True)

        # Spawn regular cars
        car_models = ['vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.citroen.c3']
        regular_cars_1, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_1, autopilot=True)
        regular_cars_2, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_2, autopilot=True)

        # Spawn pedestrians
        # pedestrians = [
        #     setup_pedestrian(world, pedestrian_spawn_point)
        #     for _ in range(1)
        # ]

        # Set spectator to focus on the AI ambulance
        # if ai_ambulance:
        #     spectator = world.get_spectator()
        #     transform = ai_ambulance.get_transform()
        #     camera_transform = carla.Transform(transform.transform(carla.Location(x=-8, z=3)), transform.rotation)  # Adjust camera position as needed
        #     spectator.set_transform(camera_transform)

        # Apply simple motion primitives only if autopilot is off
        if ai_ambulance and not ai_ambulance_autopilot:
            control = carla.VehicleControl(throttle=0.7, steer=0.0)
            ai_ambulance.apply_control(control)

        if human_ambulance and not human_ambulance_autopilot:
            control = carla.VehicleControl(throttle=0.7, steer=0.0)
            human_ambulance.apply_control(control)

        # Run the scenario for a fixed duration
        # DATA_COLLECTION = True #
        start_time = time.time()
        scenario_duration = 5
        while time.time() - start_time < scenario_duration:
            print(time.time() - start_time)
            time.sleep(1)  # assuming running in synchronous mode

    finally:
        # Clean up and reset the vehicles and pedestrians
        if ai_ambulance:
            ai_ambulance.destroy()
        if human_ambulance:
            human_ambulance.destroy()
        if regular_cars_1:
            regular_cars_1.destroy()
        if regular_cars_2:
            regular_cars_2.destroy()
        # for pedestrian in pedestrians:
        #     if pedestrian:
        #         pedestrian.destroy()

        print("Scenario ended, cleaned up the vehicles and pedestrians.")

if __name__ == '__main__':
    main()
