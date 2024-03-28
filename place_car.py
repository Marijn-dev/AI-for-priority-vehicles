# I used this to figure out where the coordinates for the map were
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(120) #Enable longer wait time in case computer is slow

world = client.get_world()
car_filter='*mini*'
vehicle_bp = world.get_blueprint_library().filter(car_filter)

inp=''
prompt_p1=' Type [a] to enable autopilot on all cars \n Type [c] to change car type \n'
prompt_p2= ' Type [@] to spawn a car at the spectator \n Type [p] to print the current spectator location \n Type [exit] to exit\n'
input_prompt=prompt_p1+prompt_p2 #the prompt is split in two for readability

vehicle_list=[]; #Save all the spawned vehicles in this list for later use
n_vehicles=0 #save the number of spawned vehicles

while inp != "exit": 
    inp = input(input_prompt)
    spectator = world.get_spectator()

    if inp== 'a':
        for i in range(n_vehicles):
            vehicle_list[i].set_autopilot(True)

    if inp== 'p':
        spec_trans=spectator.get_transform()
        print('x='+ str(spec_trans.location.x)+' y='+str(spec_trans.location.y)+' z='+str(spec_trans.location.z))

    if inp == '@':
        point=spectator.get_transform()
        vehicle_list.append(world.try_spawn_actor(vehicle_bp[0],point))
        n_vehicles = n_vehicles+1

    if inp == 'c':
        inp3=input('Type blueprint ID as [mini] check /CarlaEU4/Content/Carla/Blueprints/Vehicles for options') 
        car_filter= '*'+inp3+'*'
        vehicle_bp = world.get_blueprint_library().filter(car_filter)
    
