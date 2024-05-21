import numpy as np
import math
import torch 
import matplotlib.pyplot as plt
import csv
import time
import MapToGrid as m2g
import LSTM as lstm
from model_load import SimpleRNN
import model_load
import torch


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

    scale_factor=100
    normalized = ((R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1))*1000*scale_factor

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
    map.fill(-10)

    # Populate the grid
    for xi, zi, cost in zip(x_indices, z_indices, cost_data):
      #  if map.shape[0] > xi >= 0 and map.shape[1] > zi >= 0:  # Ensure indices are within the map bounds
        map[xi, zi] = max(map[xi, zi], cost)  # Safely use max on single elements

    return map

def trim_active_set(x,z):
    condition=((x>60) | (x<-60) | (z<0) | (z>120))
    x[condition] =0
    #y[condition] =0
    z[condition] =0
    return x,z

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

def get_new_positions(participant):
    location=participant.get_transform().location
    return location[0],location[2]
      


def __main__():
    #get image from simulation not from file
    client = carla.Client('localhost', 2000)
    client.set_timeout(120) #Enable longer wait time in case computer is slow

    world = client.get_world()

    #Run scenario and import knowledge from carla.

    depth_data=plt.imread('Rubens test files/Pictures/depth_camera_Sun_Apr_14_20_33_08_2024.png') #to get the data as an array
    segment_data=plt.imread('Rubens test files/Pictures/instance_camera_Sun_Apr_14_20_33_08_2024.png') #to get the data as an array
    
    depth_data=convert_image_to_depth(depth_data)
    segment_data=np.round(segment_data*255)
    labels=segment_data[:,:,0]

    depth_Width=depth_data[1,:,0].size
    depth_Height=depth_data[:,1,0].size

    
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
    active_set_x.append(x)
    active_set_y.append(y)
    active_set_z.append(z)

    #get data from the other participants
    #load model
    # hidden_size = 250
    # batch_size = 1
    # num_layers = 3
    future_timesegments = 2

    model_name = 'RNN_PAST5_FUTURE' + str(future_timesegments) + '.pt'
    rnn_model = torch.load('models/' + model_name)

    #gather data from carla
    #initialize
    #in vehicle or pedestrian out new data points 
    participants=[] #get these from scenario initialization |carla objects list
    ambulance= [] # define this in the scenario
    past = np.zeros(2,5) #tensor two coordinates five timeframes
    hidden=[]
    scenario_duration = 30

    prediction_horizon=5
    
    #append 
    previous_ambulance_location = ambulance_location
    previous_ambulance_rotation = ambulance_rotation
    ambulance_location = ambulance.get_transform().location
    ambulance_rotation = ambulance.get_transform().rotation


    #calculate new position of the vehicle
    veh_angle = previous_ambulance_rotation[0] #Pitch in carla's coordinate system
    x_move = np.cos(veh_angle)*(previous_ambulance_location[0]-ambulance_location[0]) - np.sin(veh_angle)*(previous_ambulance_location[2]-ambulance_location[2])
    z_move = np.sin(veh_angle)*(previous_ambulance_location[0]-ambulance_location[0]) + np.cos(veh_angle)*(previous_ambulance_location[2]-ambulance_location[2])
   
    active_set_x,active_set_z=translate_active_set(active_set_x,active_set_z,x_move,z_move,veh_angle)
    active_set_x,active_set_z=trim_active_set(active_set_x,active_set_z)
    #execute prediction

    
    hidden = rnn_model.init_hidden(past)
    future_pred, _ = rnn_model(past,hidden)

    #create label array and fuse pedestrians to vehicles

    participants_labels=[todo]
    participants_positions=[todo]
    #make a collission map
        
        #Add new coordinates to the past data
    new_coords=np.zeros(len(participants)*2,0)
    for p in participants:
        new_coords[2*p:2*p+1]=get_new_positions(p)
        past.prepend(new_coords)
        past=np.delete(past,5,0)

        #get predictions
        for p in range(participants):
            input = torch.tensor(past[p*2:p*2+1,:])
            pred = model_load.prediction(rnn_model,input,future_timesegments)
            pred.detach().numpy().prepend(participants_positions[2*p:2*p+1,:])
    
        # sample_input = torch.tensor([[[ 9.9623, 40.1296],
        #                             [17.9857, 40.0803],
        #                             [25.6637, 40.1170],
        #                             [32.3959, 40.1765],
        #                             [38.2145, 40.2339]]])
        #apply predictions to the maps
        collision_map=m2g.create_collision_map(participants_labels,pred,cost_map,prediction_horizon)
        
        




        
        
    #Test map



    
