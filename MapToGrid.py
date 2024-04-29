import numpy as np

#Initiate the map
def initiate_map(width: int,height: int,cell_size=0.1):
    '''
    Initiates the costmap width and height divided by cellsize need to be integers 
    ---------------------
    Parameters:
    width: the width of the map in meters
    height: the height of the map in meters 
    cell_size: the size of the cells in meters 
    '''
    
    return costmap

def map2grid(map,x,z,labels,width,height,cell_size: float =0.1):
    '''
    Transform the data into the gridded map
    
    ----------------------------
    Parameters:
    width: the width of the map in meters
    height: the height of the map in meters 
    cell_size: the size of the cells in meters 

    data as a [width,height,[x,y,label]]
    '''
    cost_data=labels2cost(labels)
    x_data= np.round(x,int(np.log10(1/cell_size)))
    z_data= np.round(z,int(np.log10(1/cell_size)))
    map=np.zeros([width,height])

    # Pre-filter data for each cell
    filtered_data = {}
    for i in range(width):
        for j in range(height):
            filtered_data[(i, j)] = cost_data[(x_data==i*cell_size) & (z_data==j*cell_size)]

    # Calculate maximum value for each cell
    for i in range(width):
        for j in range(height):
            if len(filtered_data[(i, j)]) > 0:
                map[i,j] = np.max(filtered_data[(i, j)])  

    return map


import numpy as np

def map3grid(map, x, z, labels, width, height, cell_size=0.1):
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
    cost_data = labels2cost(labels)  # Assuming labels2cost is defined elsewhere

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
    map.fill(0)

    # Populate the grid
    for xi, zi, cost in zip(x_indices, z_indices, cost_data):
        if map.shape[0] > xi >= 0 and map.shape[1] > zi >= 0:  # Ensure indices are within the map bounds
            map[xi, zi] = max(map[xi, zi], cost)  # Safely use max on single elements


    return map
    


def labels2cost(labels):      #for more info on the labels visit: https://carla.readthedocs.io/en/latest/ref_sensors/#instance-segmentation-camera
    specific_costs = {0:0,    #unlabeld things and things without collissions
                      1:0,    #roads 
                      2:50,   #sidewalks
                      3:255,  #buildings
                      4:254,  #Wall
                      5:253,  #Fence
                      6:252,  #Pole
                      7:252,  #Traffic Lights (Always comes with a pole) 
                      8:252,  #Traffic Sign
                      9:251,  #Vegetation (Trees bushes etc)
                      10:10,  #Terrain (Ground level vegetation)
                      11:0,   #Sky
                      12:256, #Pedestrian
                      13:256, #Rider (Bikers etc)
                      14:256, #Cars
                      15:256, #Trucks
                      16:256, #Busses
                      17:256, #Train
                      18:256, #motorcycle
                      19:256, #Bicycle
                      20:250, #static objects (fire hydrants, busstops, etc)
                      21:256, #Movable trash bins, buggies, bags, wheelchairs, animals, etc.
                      22:256, #Other If we dont know what is, better not hit it
                      23:250, #Water
                      24:0,   #Roadlines
                      25:0,   #Ground but not road or vegitation
                      26:150, #The structure of bridges not the surfaces
                      27:150, #rail tracks without proper car crossings
                      28:249, #Guard rail
                     'default': 255, # Add a default cost for unrecognized labels
                      }
    # return np.vectorize(specific_costs.get)(labels)
    return np.vectorize(lambda x: specific_costs.get(x, specific_costs['default']))(labels)