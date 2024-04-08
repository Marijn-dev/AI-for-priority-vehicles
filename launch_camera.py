#setup etc
import carla
import numpy as np
#import pygame
import cv2
try:
   import queue
except ImportError:
   import Queue as queue
# # Render object to keep and pass the PyGame surface
# class RenderObject(object):
#     def __init__(self, width, height):
#         init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
#         self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# # Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
# def pygame_callback(data, obj):
#     img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
#     img = img[:,:,:3]
#     img = img[:, :, ::-1]
#     obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))


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


#add an instance segmentation camera
instance_segmentation_camera= blueprint_library.find('sensor.camera.instance_segmentation')
camera_init_trans = carla.Transform(carla.Location(z=1.5)) 
camera = world.try_spawn_actor(instance_segmentation_camera, camera_init_trans, attach_to=ego_vehicle)



input("press enter when ready")
ego_vehicle.set_autopilot(True)

instance_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
# Now we have to spawn the camera on the car (attach_to=vehicle), x and y are relative locations and will differ per vehicle
spawn_point = carla.Transform(carla.Location(x=2.5, z=1.7))
instance_camera = world.spawn_actor(instance_camera_bp, spawn_point, attach_to=ego_vehicle)


instance_image_queue = queue.Queue()
instance_camera.listen(instance_image_queue.put)
instance_image=instance_image_queue.get()
instance_image.save_to_disk('instance_segmentation.png')

a = np.array(instance_image.raw_data)
print(a)
print(np.shape(a))
print(np.unique(a))

print(instance_image)

input('Done?')
every_actor = world.get_actors()
for sensor in every_actor.filter('sensor.*'):
    print(sensor)
    sensor.destroy()
    
for vehicle in every_actor.filter('vehicle.*'):
    print(vehicle)
    print(actor_list[0])
    vehicle.destroy()

