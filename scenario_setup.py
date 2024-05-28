import carla
import random
import time


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
    spawn_location = carla.Location(x=232.24923706054688+random.randrange(-20, 20,2), y=-310.7073059082031, z=1.3423326015472412)
    spawn_rotation = carla.Rotation(pitch=0, yaw=178.68557739257812, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_spawn_point_4(world):
    spawn_location = carla.Location(x=175.24923706054688+random.randrange(-20, 20,2), y=-307.7073059082031, z=1.3423326015472412)
    spawn_rotation = carla.Rotation(pitch=-6, yaw=-1.1, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

import carla
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

def scenario_setup():
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05  # Set a fixed time-step
    settings.synchronous_mode = True  # Enable synchronous mode

    # Configure substepping parameters
    settings.max_substep_delta_time = 0.01  # Maximum time step for a physics substep
    settings.max_substeps = 10  # Maximum number of physics substeps

    if settings.fixed_delta_seconds > settings.max_substep_delta_time * settings.max_substeps:
        raise ValueError("fixed_delta_seconds must be <= max_substep_delta_time * max_substeps")

    world.apply_settings(settings)

    if world.get_map().name == "Carla/Maps/Town10HD_Opt":
        world = client.load_world('Town04')
    
    actor_list = world.get_actors()
    for a in actor_list:  # Removes old actors so you can just run the script again and again
        if a.id > 146:
            a.destroy()

    spawn_point_pool = [find_spawn_point_1(world), find_spawn_point_2(world), find_spawn_point_3(world), find_spawn_point_4(world)]
    random.shuffle(spawn_point_pool)
    
    ai_ambulance_spawn_point = spawn_point_pool.pop()
    ambulance_spawn_point = spawn_point_pool.pop()
    car_spawn_point_1 = spawn_point_pool.pop()
    car_spawn_point_2 = spawn_point_pool.pop()

    car_models = ['vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.citroen.c3']
    regular_cars_1, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_1, autopilot=False)
    regular_cars_2, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_2, autopilot=False)
    
    ai_ambulance, ai_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ai_ambulance_spawn_point)
    participants = [regular_cars_1, regular_cars_2]
    participant_labels = ["car", "car"]

    camera_transform = carla.Transform(carla.Location(x=3.5, z=1.0))
    blueprint = world.get_blueprint_library().find('sensor.camera.depth')
    depth_camera = world.spawn_actor(blueprint, camera_transform, attach_to=ai_ambulance)

    seg_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
    segment_camera = world.spawn_actor(seg_blueprint, camera_transform, attach_to=ai_ambulance)

    # Set the spectator position
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        carla.Location(x=185.65426635742188, y=-328.8107604980469, z=33.317054748535156),
        carla.Rotation(pitch=-50.184872, yaw=41.805153, roll=0.000003)
    )
    spectator.set_transform(spectator_transform)

    world.tick()  # Synchronize the world state

    # Apply a simple motion primitive to the regular cars (example: go straight)
    if regular_cars_1:
        control = carla.VehicleControl(throttle=0.7, steer=0.0)
        regular_cars_1.apply_control(control)

    if regular_cars_2:
        control = carla.VehicleControl(throttle=0.7, steer=0.0)
        regular_cars_2.apply_control(control)

    world.tick()  # Ensure the controls are applied in the same tick

    return ai_ambulance, participants, participant_labels, depth_camera, segment_camera, world

