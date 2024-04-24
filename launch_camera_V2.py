#setup etc
import carla
import numpy as np
#import pygame
import cv2
import time
try:
   import queue
except ImportError:
   import Queue as queue


def camera_callback(image, data):
    capture = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    data['image'] = capture
    cv2.imwrite(f"{image.frame}.png", capture)

def process_img(data):
    i = np.array(data.raw_data)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) # ,4 cause its of the a in rgba
    i3 = i2[:,:,:3] # only interested in first 3 channels (RGB)

    # show realtime images
    cv2.imshow("",i3)
    cv2.waitKey(1)

    return i3/255.0

client = carla.Client('localhost', 2000)
client.set_timeout(120) #Enable longer wait time in case computer is slow

world = client.get_world()
car_filter='*Ambulance*' 
vehicle_bp = world.get_blueprint_library().filter(car_filter)
blueprint_library= world.get_blueprint_library()

#Spawn a car at the spectator
spectator = world.get_spectator()
point=spectator.get_transform()
ego_vehicle=world.try_spawn_actor(vehicle_bp[0],point)
actor_list = []

input("press enter when ready")
ego_vehicle.set_autopilot(True)

# instance_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
# # Now we have to spawn the camera on the car (attach_to=vehicle), x and y are relative locations and will differ per vehicle
# spawn_point = carla.Transform(carla.Location(x=2.5, z=1.7))
# instance_camera = world.spawn_actor(instance_camera_bp, spawn_point, attach_to=ego_vehicle)

depth_camera_bp= blueprint_library.find('sensor.camera.depth')
camera_init_trans = carla.Transform(carla.Location(z=1.5)) 
depth_camera = world.try_spawn_actor(depth_camera_bp, camera_init_trans, attach_to=ego_vehicle)

depth_image_queue = queue.Queue()
instance_image_queue = queue.Queue()

#Create a loop to allow the user to take pictures
while input('take a picture or [Exit]?')!='Exit':
    string=time.ctime().replace(" ","_").replace(":","_") #create a unique part of the file name for different pictures
    
    # instance_camera.listen(instance_image_queue.put)
    # instance_image=instance_image_queue.get()
    # instance_image.save_to_disk(r"C:\Users\20192709\Documents\5IAP0 Interdisciplinary team project\Pictures\instance_camera_"+string +".png")

    depth_camera.listen(depth_image_queue.put)
    depth_image=depth_image_queue.get()
    depth_image.save_to_disk(r"C:\Users\20192709\Documents\5IAP0 Interdisciplinary team project\Pictures\depth_camera_"+string +".png")

#Destroy the sensors and cars to clean up
every_actor = world.get_actors()
for sensor in every_actor.filter('sensor.*'):
    sensor.destroy()

for vehicle in every_actor.filter('vehicle.*'):
    print(actor_list[0])
    vehicle.destroy()