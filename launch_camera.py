#setup etc
import carla
import numpy as np
import pygame

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))


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



#add an instance segmentation camera
instance_segmentation_camera= blueprint_library.find('sensor.camera.instance_segmentation')
camera_init_trans = carla.Transform(carla.Location(z=1.5)) 
camera = world.try_spawn_actor(instance_segmentation_camera, camera_init_trans, attach_to=ego_vehicle)



input("wait")
camera.listen(lambda image: pygame_callback(image, RenderObject))
ego_vehicle.set_autopilot(True)