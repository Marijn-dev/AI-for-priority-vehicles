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

actors = world.get_actors()
vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]

i = 0
while i < 10:
    actors = world.get_actors()
    vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]
    for vehicle in vehicles:
        print(vehicle.type_id)
    print(i)
    time.sleep(5)
    i = i + 1