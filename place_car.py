# I used this to figure out where the coordinates for the map were
import carla



client = carla.Client('localhost', 2000)
client.set_timeout(120) #Enable longer wait time in case computer is slow

x_spawn=0
y_spawn=0
z_spawn=1 #dont spawn at z=0 otherwise it falls through the map

world = client.get_world()
vehicle_bp = world.get_blueprint_library().filter('*mini*')

inp=''


while inp != "exit":
    inp = input('to change spawn coordinate type coordinate [x,y,z],to spawn car type [s], type a to enable autopilot, p to print the current spawn point or type exit to exit')
    if inp == 'x':
        inp2= input('Increase x by how much?')
        x_spawn=x_spawn+float(inp2)

    if inp == 'y':
        inp2= input('Increase y by how much?')
        y_spawn=y_spawn+float(inp2)

    if inp == 'z':
        inp2= input('Increase z by how much?')
        z_spawn=z_spawn+float(inp2)

    if inp == 's':
        point= carla.Transform(location=carla.Location(x=x_spawn,y=y_spawn,z=z_spawn),rotation=carla.Rotation(0,0,0))
        vehicle1 = world.try_spawn_actor(vehicle_bp[0],point)
        spectator = world.get_spectator()
        spec_point=point
        spec_point.location.z = point.location.z+3 #start_point was used to spawn the car but we move 1m up to avoid being on the floor
        spectator.set_transform(spec_point)
    if inp== 'a':
        vehicle1.set_autopilot(True)
    if inp== 'p':
        print(x_spawn,y_spawn,z_spawn)