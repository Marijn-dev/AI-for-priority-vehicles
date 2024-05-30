import numpy as np
import math
import torch 
import matplotlib.pyplot as plt
import csv
import time
import MapToGrid as m2g
import LSTM as lstm
import queue
import torch

import carla
from LSTM_predict import SimpleRNN
import LSTM_predict
import scenario_setup as scene
import selection_motion_primitive as mp


# Define the target coordinates for each spawn point
starting_target_pairs = {
    "find_spawn_point_1": (200, -278), # town
    "find_spawn_point_2": (235, -307), # side town
    "find_spawn_point_3": (166, -311), # houses
    "find_spawn_point_4": (205, -341) # highway
}

def get_point_image(point_img,K_inv,Width,Height):
        loc_mat=np.zeros([Height,Width,3])
        static_projection_matrix=np.zeros([Height,Width,3])
        for i in range(Height):
                for j in range(Width):
                        loc_mat[i,j] = [j-Width/2,i-Height/2,1]
                        static_projection_matrix[i,j]=np.dot(K_inv,loc_mat[i,j])
        return static_projection_matrix

def convert_image_to_depth(image_data):
    R=image_data[:,:,0]
    G=image_data[:,:,1]
    B=image_data[:,:,2]

    scale_factor=256
    normalized = ((R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1))*1000*scale_factor
    return normalized

def define_camera_matrix():
    fov = 90
    im_size_x = 800
    im_size_y = 600

    f  = im_size_x /(2.0 * math.tan(fov * math.pi / 360))
    Cx = im_size_x / 2.0
    Cy = im_size_y / 2.0

    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(K)

def calc_cartesian_image_data(rel_coords,depth_values):
    '''
    Figure out where the pixels in the image are relative to the car.
    This is using properties of similar triangles to calculate the real x,y and z
    -----------------------------
    Parameters:
    rel_coords = a matrix of (image_width,image_height,[x,y,1]) 
    depth_values= a matrix of (image_width,image_height,depth_value)    
    
    '''
    #we use np.division to quickly do elementwise matrix division where dividing by 0 equals 1
    focal_mat=np.ones_like(rel_coords[:,:,0])
    camera_index_mat=rel_coords

    c_squared=np.square(focal_mat)+np.square(camera_index_mat[:,:,1])
    d=np.power(np.square(camera_index_mat[:,:,0])+c_squared, 0.5*np.ones_like(camera_index_mat[:,:,0]))

    #Create a condition to bypass the division incase of zeros
    condition= (np.square(camera_index_mat[:,:,0]) + np.square(camera_index_mat[:,:,1]))!=0

    #cant put a scalar in np.divide so we use ones
    placeholder=np.ones_like(camera_index_mat[:,:,0])

    a  = np.divide(placeholder, d, out=np.zeros_like(placeholder), where= condition)

    x= a*depth_values*(camera_index_mat[:,:,0]+1)
    y= a*depth_values*(camera_index_mat[:,:,1]+.75)
    z= np.sqrt(np.power(depth_values,2)-np.power(x,2)-np.power(y,2)) #This loses some values but i cannot figure out why.
    return x,y,z

def filter_data(x,y,z):
    x[np.isnan(x)==True]=0
    y[np.isnan(y)==True]=0  
    z[np.isnan(z)==True]=0
    return x,y,z

def map2grid(map, x, z, labels, width, height, cell_size=0.1):
    """
    Transform the data into the gridded map.

    Parameters:
    map (np.array): The initial costmap grid.
    x (np.array): The x coordinates of the data points.
    z (np.array): The z coordinates of the data points.
    labels (np.array): The labels associated with each data point.
    width (float): The width of the map in meters.
    height (float): The height of the map in meters.
    cell_size (float): The size of the cells in meters.
    """
    cost_data = m2g.labels2cost(labels)  # Assuming labels2cost is defined elsewhere

    # Adjust coordinates to start at the middle bottom of the grid
    x_offset = width / 2
    x_centered = x + x_offset
    z_centered = z  # Assuming z starts at 0 at the bottom

    # Convert to grid indices
    x_indices = np.clip((x_centered / cell_size).astype(int), 0, int(width / cell_size) - 1)
    z_indices = np.clip((z_centered / cell_size).astype(int), 0, int(height / cell_size) - 1)

    # Flatten the data if needed
    x_indices = x_indices.flatten()
    z_indices = z_indices.flatten()
    cost_data = cost_data.flatten()


    # Clear the map for new data
    #map.fill(-10)

    # Populate the grid
    for xi, zi, cost in zip(x_indices, z_indices, cost_data):
      #  if map.shape[0] > xi >= 0 and map.shape[1] > zi >= 0:  # Ensure indices are within the map bounds
        map[xi, zi] = max(map[xi, zi], cost)  # Safely use max on single elements

    return map

def trim_active_set(x,y,z):
    y=np.array(y)
    condition=((x>60) | (x<-60) |(y<1) | (z<0) | (z>120)) #note that the sensor is already on top of the car so the 0,0 point is not at floor level
    x[condition]=0
    y[condition]=0
    z[condition]=0
    return x,y,z

def translate_active_set(x_data,z_data,x_move,z_move,theta):
    #rotation of a frame just along the y axis https://nl.mathworks.com/help/fusion/gs/spatial-representation-coordinate-systems-and-conventions.html
    x_data= (x_data+ x_move)*np.cos(theta) + (z_data+ z_move)*-np.sin(theta)
    z_data= (x_data+ x_move)*np.sin(theta) + (z_data+ z_move)*np.cos(theta)
    return x_data,z_data
     
def write_relative_positions(writer, timestamp, actor_type, actors, reference_location):
    for actor in actors:
        actor_location = actor.get_transform().location
        relative_x = actor_location.x - reference_location.x
        relative_y = actor_location.y - reference_location.y
        writer.writerow([timestamp, actor_type, actor.id, relative_x, relative_y])

def get_new_positions(participant,ambulance):
    location=participant.get_transform().location - ambulance.get_transform().location
    theta=ambulance.get_transform().rotation.yaw
    location.x= (location.x)*np.cos(theta) + (location.x)*-np.sin(theta)
    location.z= (location.x)*np.sin(theta) + (location.z)*np.cos(theta)
    return location.x,location.z
      
def create_collision_map(participants_labels,participants_positions,cost_map,prediction_horizon):
    #ASsumes the positions are in the grid size already
    #create a matrix per timestep M(x,z,t)
    M=[np.zeros_like(cost_map)]*prediction_horizon

    for p in range(len(participants_labels)):
        for t in range(len(participants_positions[:,0])):
            M=place_traffic_participants(t,p,participants_labels,participants_positions,M)
    return M

   
   
   

def place_traffic_participants(t,p,participants_labels,participants_positions,M,cell_multiplier=10):  
    #Known issue: this places high costs in the corners, fix later as it is not relevant for its function.
    #coordinates relative to the ego car
    x_car=int(np.round(participants_positions[t,p*2]*cell_multiplier))
    z_car=int(np.round(participants_positions[t,p*2+1]*cell_multiplier)+600)
    
    if participants_labels[p] =='car':
        actor_radius = 10 #in gridpoints (assumption assuming gridsize=0.1m)
    
    if participants_labels[p] =='pedestrian':
        actor_radius = 5 #in gridpoints (assumption assuming gridsize=0.1m)
    
    for x in range(-actor_radius + x_car,actor_radius + x_car,1):
        for z in range(-actor_radius + z_car,actor_radius + z_car,1):
            M[t][x,z]=512
    return M

def get_primitives():
    primitives = [
        {'curvature': 0, 'distance': 10, 'velocity': 0.5},
        {'curvature': 0, 'distance': 10, 'velocity': 0.65},
        {'curvature': 0, 'distance': 20, 'velocity': 0.75},
        {'curvature': 0, 'distance': 20, 'velocity': 0.9},
        {'curvature': 2.5, 'distance': 10, 'velocity': 0.5},
        {'curvature': 2.5, 'distance': 20, 'velocity': 0.7},
        {'curvature': 5, 'distance': 10, 'velocity': 0.5},
        {'curvature': 5, 'distance': 20, 'velocity': 0.7},
        {'curvature': 10, 'distance': 10, 'velocity': 0.5},
        {'curvature': 10, 'distance': 10, 'velocity': 0.7},
        {'curvature': 15, 'distance': 10, 'velocity': 0.5},
        {'curvature': 15, 'distance': 10, 'velocity': 0.7},
        {'curvature': -2.5, 'distance': 10, 'velocity': 0.5},
        {'curvature': -2.5, 'distance': 20, 'velocity': 0.7},
        {'curvature': -5, 'distance': 10, 'velocity': 0.5},
        {'curvature': -5, 'distance': 20, 'velocity': 0.7},
        {'curvature': -10, 'distance': 10, 'velocity': 0.5},
        {'curvature': -10, 'distance': 10, 'velocity': 0.7},
        {'curvature': -15, 'distance': 10, 'velocity': 0.5},
        {'curvature': -15, 'distance': 10, 'velocity': 0.7},
    ]
    return primitives



def main():   
    should_print=True 
    #Run scenario and import knowledge from carla.
    
    # Define the goal target key for the ambulance
    goal_target = 'find_spawn_point_1'  # Options: 'find_spawn_point_1', 'find_spawn_point_2', 'find_spawn_point_3', 'find_spawn_point_4'
    

    ambulance, participants, participants_labels,depth_camera,segment_camera,world,target =scene.scenario_setup(goal_target)
    ambulance_location=ambulance.get_transform().location
    ambulance_rotation=ambulance.get_transform().rotation 

    # depth_data=plt.imread('AI-for-priority-vehicles\Rubens_test_files\Pictures\depth_camera_Sun_Apr_14_20_33_08_2024.png') #to get the data as an array
    # segment_data=plt.imread('AI-for-priority-vehicles\Rubens_test_files\Pictures\instance_camera_Sun_Apr_14_20_33_08_2024.png') #to get the data as an array
    
    depth_data=plt.imread('/home/zyd/AI-for-priority-vehicles/Rubens_test_files/Pictures/depth_004641.png') #to get the data as an array
    segment_data=plt.imread('/home/zyd/AI-for-priority-vehicles/Rubens_test_files/Pictures/instance_004641.png') #to get the data as an array
    


    depth_data=convert_image_to_depth(depth_data)
    segment_data=np.round(segment_data*255)
    labels=segment_data[:,:,0]
   
    depth_Width=len(depth_data[1,:])
    depth_Height=len(depth_data[:,1])
    
    K_inv=define_camera_matrix()
    camera_coordinates=get_point_image(depth_data,K_inv,depth_Width,depth_Height)

    x,y,z = calc_cartesian_image_data(camera_coordinates,depth_data)

    x,y,z = filter_data(x,y,z)

    map_width=120 #meters
    map_height=240 #meters
    cell_size=0.1 #meters
    

    active_set_x=[]
    active_set_y=[]
    active_set_z=[]

    cost_map=np.zeros([int(map_width/cell_size),int(map_height/cell_size)])

    #split into loop
    #add new data to the cost_map
    cost_map = map2grid(cost_map, x, z, labels, map_width, map_height, cell_size)
    #save the points


    #get data from the other participants
    #load model
    # hidden_size = 250
    # batch_size = 1
    # num_layers = 3
    past_timefragments=5
    future_timesegments = 4
      
    #gather data from carla
    #initialize
    #in vehicle or pedestrian out new data points 
    past = np.zeros([past_timefragments,2*len(participants)])
    hidden=[]
    scenario_duration = 30
    prediction_horizon= future_timesegments
    
    # participants_positions=np.zeros(len(participants)*2)
    # i=0
    # for p in participants:
    #     participants_positions[2*i] =p.get_transform().location.x
    #     participants_positions[2*i+1] = p.get_transform().location.z
    #     i+=1
    

        

        
    ##################################################
    #Loop
    i_time_steps=0
    step_time=0.5
    n_steps=30
    start_time = time.time()
    start = time.time()
    while i_time_steps<n_steps:
        i_time_steps += 1
        world.tick()

        time.sleep(step_time - 0.2)

        # while time.time() < (start + step_time * i_time_steps):
        #     pass

        #redefine camera to change segmentation error
        depth_camera.destroy()
        segment_camera.destroy()
        camera_transform = carla.Transform(carla.Location(x=3.5, z=1.0))
        blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera = world.spawn_actor(blueprint, camera_transform, attach_to=ambulance)

        seg_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        segment_camera = world.spawn_actor(seg_blueprint, camera_transform, attach_to=ambulance)

        image_queue = queue.LifoQueue()
        depth_camera.listen(lambda data: image_queue.put(data))
        depth_data = image_queue.get().raw_data
        depth_data=np.reshape(depth_data,[600,800,4])
        
        depth_data=convert_image_to_depth(depth_data)
        
        seg_image_queue = queue.LifoQueue()
        segment_camera.listen(lambda data: seg_image_queue.put(data))
        segment_data = seg_image_queue.get().raw_data
        segment_data=np.reshape(segment_data,[600,800,4])
        segment_data=np.round(segment_data*255)
        labels=segment_data[:,:,0]

        previous_ambulance_location = ambulance_location
        previous_ambulance_rotation = ambulance_rotation
        ambulance_location = ambulance.get_transform().location
        ambulance_rotation = ambulance.get_transform().rotation

        new_coords=np.zeros(len(participants)*2)
        for p in range(len(participants)):
            new_coords[2*p],new_coords[2*p+1]=get_new_positions(participants[p],ambulance)
        past= np.vstack((past,new_coords))
        past= np.delete(past,0,0) 
            #transform coordinates to local in the participants views
        local_par_pos = np.zeros_like(past)
        for c in range(len(past)-1):
            local_par_pos[:,c]= past[:,c]-past[0,c] #subtract the earliest coordinate as the trajectories need to start at 0 for the model
        
        predictions=np.zeros([future_timesegments,2*len(participants)])
        for p in range(len(participants)):
            input = torch.tensor([local_par_pos[:,p:p+2]]).to(torch.float32)
            pred = LSTM_predict.prediction(input)
            predictions[:,2*p:2*p+2]=pred.detach().numpy()[0]

        veh_angle = previous_ambulance_rotation.yaw-ambulance_rotation.yaw 
        x_move = np.cos(veh_angle)*(previous_ambulance_location.x-ambulance_location.x) - np.sin(veh_angle)*(previous_ambulance_location.z-ambulance_location.z)
        z_move = np.sin(veh_angle)*(previous_ambulance_location.x-ambulance_location.x) + np.cos(veh_angle)*(previous_ambulance_location.z-ambulance_location.z)
    
        
        #Take pictures
        
        
        x,y,z = calc_cartesian_image_data(camera_coordinates,depth_data)
        x,y,z = filter_data(x,y,z)

        active_set_x,active_set_z=translate_active_set(active_set_x,active_set_z,x_move,z_move,veh_angle)

        active_set_x = np.append(active_set_x,x)
        active_set_y = np.append(active_set_y,y)
        active_set_z = np.append(active_set_z,z)

        active_set_x,active_set_y,active_set_z=trim_active_set(active_set_x,active_set_y,active_set_z)


        cost_map = map2grid(cost_map, active_set_x, active_set_z, labels, map_width, map_height, cell_size)
        
        collision_map=create_collision_map(participants_labels,predictions,cost_map,prediction_horizon)

        #Calculate 
        primitives=get_primitives()
        costs = mp.calculate_primitive_costs(cost_map, collision_map, primitives, cell_size, 2.4, ambulance_location, target)
        best_primitive = mp.select_best_primitive(costs)
        throttle,steer,brake = mp.convert_to_vehicle_control(best_primitive)
        ambulance.apply_control(carla.VehicleControl(throttle,steer,brake))

        #Report
        print(time.time())
        print(best_primitive)
        if should_print and time.time()-start>15 :
            should_print=False





if __name__ == "__main__":
    main()