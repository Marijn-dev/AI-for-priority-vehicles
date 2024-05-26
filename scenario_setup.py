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
    settings.fixed_delta_seconds = 1 # Set a variable time-step
    world.apply_settings(settings)


    if world.get_map().name =="Carla/Maps/Town10HD_Opt":
        world = client.load_world('Town04')
    actor_list = world.get_actors()
    for a in actor_list: #removes old actors so you can just run the script again and again
        if a.id>146:
            a.destroy()
    
    spawn_point_pool = [find_spawn_point_1(world), find_spawn_point_2(world), find_spawn_point_3(world), find_spawn_point_4(world)]

    ai_ambulance_spawn_point = spawn_point_pool.pop()
    ambulance_spawn_point = spawn_point_pool.pop()
    car_spawn_point_1 = spawn_point_pool.pop()
    car_spawn_point_2 = spawn_point_pool.pop()

    car_models = ['vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.citroen.c3']
    regular_cars_1, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_1)
    regular_cars_2, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_2)
    

    ai_ambulance, ai_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ai_ambulance_spawn_point)
    participants = [regular_cars_1,regular_cars_2]
    participant_labels = ["car","car"]
    return ai_ambulance ,participants,participant_labels