import carla
import math
import random
import time
import numpy as np
import cv2
import time
try:
   import queue
except ImportError:
   import Queue as queue

client = carla.Client('localhost', 2000) # port 2000
client.set_timeout(10.0) # seconds
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = False
# settings.fixed_delta_seconds = 0.02
world.apply_settings(settings) 

world = client.load_world('Town01')


## Spawn cars

blueprint_library = world.get_blueprint_library()
model3 = blueprint_library.filter('model3')[0]
ambulance = blueprint_library.filter('vehicle.ford.ambulance')[0]
spawn_point = carla.Transform(carla.Location(x=88.619987, y=101.833946, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90.000046, roll=0.000000)) 
spawn_point2 = carla.Transform(carla.Location(x=88.619987, y=101.833946+12, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90.000046, roll=0.000000)) 
spawn_point3 = carla.Transform(carla.Location(x=88.619987+4, y=101.833946+12, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90.000046, roll=0.000000)) 

vehicle1 = world.spawn_actor(ambulance, spawn_point) # vehicle 1
vehicle2 = world.spawn_actor(model3, spawn_point2) # vehicle 2
vehicle3 = world.spawn_actor(model3, spawn_point3) # vehicle 2

spectator = world.get_spectator()
transform = carla.Transform(vehicle1.get_transform().transform(carla.Location(x=+4,z=2.5)), vehicle1.get_transform().rotation)
spectator.set_transform(transform)

# time.sleep

# apply input to other traffic
actors = world.get_actors()

vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]

# make cars drive
for vehicle in vehicles:
    if "ambulance" not in vehicle.type_id:
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0, brake=0))

# wait 10 secs
time.sleep(10)

# stop cars
for vehicle in vehicles:
    if "ambulance" not in vehicle.type_id:
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

# destroy all cars
time.sleep(5)
# # destroy vehicles
# for vehicle in vehicles:
#     vehicle.destroy()