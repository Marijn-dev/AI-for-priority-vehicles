import numpy as np

def map2grid(data,width,height,cell_size=0.1):
    static_costmap= np.zeros(round(width/cell_size),round(height/cell_size))

    x_data= np.round(data[:,:,0],int(np.log10(1/cell_size)))
    z_data= np.round(data[:,:,1],int(np.log10(1/cell_size)))
    cost_data = data[:,:,2]

    for i in range(width):
        for j in range(height):
            static_costmap[i,j] = np.max(cost_data[x_data==x_data[i] and z_data==z_data[i]])


def select_information(data,width,height):
    