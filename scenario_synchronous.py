import sys
import glob
import os

try:
    sys.path.append(glob.glob('/home/marijn/Downloads/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import time
import random
import csv
import numpy as np
import cv2
import queue
import threading

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
    spawn_location = carla.Location(x=232.24923706054688+random.randrange(-20, 20,2), y=-310.7073059082031, z=1.3423326015472412)
    spawn_rotation = carla.Rotation(pitch=0, yaw=178.68557739257812, roll=0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_transform

def find_spawn_point_4(world):
    spawn_location = carla.Location(x=175.24923706054688+random.randrange(-20, 20,2), y=-307.7073059082031, z=1.3423326015472412)
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

def write_relative_positions(writer, timestamp, actor_type, actors, spawn_points):
    for i, actor in enumerate(actors):
        actor_location = actor.get_transform().location
        global_x = actor_location.x - spawn_points[i].location.x
        global_y = actor_location.y - spawn_points[i].location.y
        writer.writerow([timestamp, actor_type, actor.id, global_x, global_y])

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True # Enables synchronous mode
    world.apply_settings(settings)
    settings = world.get_settings()
  
    world.apply_settings(settings)
    # Comment sentence below out if running the script for a secwond time
    world = client.load_world('Town04')
    data_runs = 1
    
    for i in range(data_runs):
        # clean up actors if any are left
        actors = world.get_actors()

        # Iterate through actors
        for actor in actors:
            # Check if actor is a vehicle
            if actor.type_id.startswith('vehicle'):
                # Destroy the vehicle
                actor.destroy()

        map = world.get_map()
        spawn_points = map.get_spawn_points()
        spawn_point_pool = [find_spawn_point_1(world), find_spawn_point_2(world), find_spawn_point_3(world), find_spawn_point_4(world)]

        # Shuffle the pool to randomize order
        random.shuffle(spawn_point_pool)
        traffic_manager = client.get_trafficmanager()
    
        # Randomly choose spawn points for each participant
        ai_ambulance_spawn_point = spawn_point_pool.pop()
        ambulance_spawn_point = spawn_point_pool.pop()
        car_spawn_point_1 = spawn_point_pool.pop()
        car_spawn_point_2 = spawn_point_pool.pop()
        pedestrian_spawn_point = find_pedestrian_spawn_point(world)


        # Spawn two ambulances
        ai_ambulance, ai_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ai_ambulance_spawn_point, autopilot=True) #color='255,0,0')
        human_ambulance, human_ambulance_autopilot = setup_vehicle(world, 'vehicle.ford.ambulance', ambulance_spawn_point, autopilot=True)

        # Spawn regular cars
        car_models = ['vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.citroen.c3']
        regular_cars_1, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_1, autopilot=True)
        regular_cars_2, _ = setup_vehicle(world, random.choice(car_models), car_spawn_point_2, autopilot=True)
        spawn_points = [ambulance_spawn_point, car_spawn_point_1, car_spawn_point_2]

        vehicles = [human_ambulance,regular_cars_1,regular_cars_2] # list of all vehicles
        
        blueprint_library = world.get_blueprint_library()
        camera_transform = carla.Transform(carla.Location(x=3.5, z=1.0))
        blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera = world.spawn_actor(blueprint, camera_transform, attach_to=ai_ambulance)

        image_queue = queue.Queue()
        
        if ai_ambulance:
            spectator = world.get_spectator()
            transform = ai_ambulance.get_transform()
            camera_transform = carla.Transform(transform.transform(carla.Location(x=-8, z=3)), transform.rotation)  # Adjust camera position as needed
            spectator.set_transform(camera_transform)

        
        
       
        depth_camera.listen(image_queue.put)
        # while True:
    
        scenario_duration = 10
        interval = 2
        steps = 0
        start = time.time()
        while time.time()-start <= scenario_duration:
            steps += 1

            time.sleep(interval - 0.7)

            while time.time() < (start + interval * steps):
                pass

            #execute your stuff
            world.tick()
            image = image_queue.get()
            print(image.frame_number)
            image.save_to_disk('images_test4/depth/%06d.png' % image.frame)
            # write_relative_positions(writer, time.time()-start, 'Vehicle',vehicles, spawn_points)
            # print(time.time(), interval * steps, time.time() - (start + (interval * steps)), flush=True)  # prints the actual interval
            print(time.time()-start)

        print("Scenario ended, cleaned up the vehicles and pedestrians.")

if __name__ == '__main__':
    main()
