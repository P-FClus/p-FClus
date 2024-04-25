import random
import numpy as np
import csv
import array as arr
import math

def gauss_2d(mu, sigma):
    # x cordinate of datapoint
    x = random.gauss(mu, sigma)
    # y cordinate of datapoit
    y = random.gauss(mu, sigma)
    return (x, y)

# homogeneity of the dataset. That is the number of gaussians that each client gets its data from.
homo = int(input("Enter Homogeneity: "))
# number of data in client 1
points_in_client1 = int(input("Enter the total data points for client 1: ")) 
# additional data on increasing the clients
additional_points_per_cluster = int(input("Enter the additional data points per cluster: "))

for _ in range(1):
    total_clusters = 20
    for _ in range(1):
        data = []
        total_clients = 1000 # Different for different dataset
        for j in range(total_clients):
            curr = []
            all_gauss = [g for g in range (total_clusters)]
            random.shuffle(all_gauss) # randomly selects a gaussian
            gauss = all_gauss[:homo] # stores selected gaussians in gauss
            for hh in range(homo):
                i = gauss[hh]
                total_points = points_in_client1+j*additional_points_per_cluster                
                from_ith_cluster_to_jth_client = math.ceil(total_points/homo)
                for n in range(from_ith_cluster_to_jth_client):
                    if i==0 or i==1:
                        x_val,y_val = gauss_2d(9*i,3)
                    elif i==2 or i==3:
                        x_val,y_val = gauss_2d(9*(i+2),3)
                    elif i==4 or i==5:
                        x_val,y_val = gauss_2d(9*(i+4),3)
                    elif i==6 or i==7:
                        x_val,y_val = gauss_2d(10*(i+6),3)
                    elif i==8 or i==9:
                        x_val,y_val = gauss_2d(10*(i+8),3)
                    elif i==10 or i==11:
                        x_val,y_val = gauss_2d(10*(i+10),3)
                    elif i==12 or i==13:
                        x_val,y_val = gauss_2d(10*(i+12),3)
                    elif i==14 or i==15:
                        x_val,y_val = gauss_2d(10*(i+14),3)
                    elif i==16 or i==17:
                        x_val,y_val = gauss_2d(10*(i+16),3)
                    elif i==18 or i==19:
                        x_val,y_val = gauss_2d(10*(i+18),3)
                    if(len(curr) == total_points):
                        break
                    else:
                        curr.append([x_val,y_val])
                        data.append([x_val,y_val]) # stores the datapoint in the data
                    
# File name for the CSV file
        file_name = f"k_{total_clusters}_c_{total_clients}_h_{homo}_p_{points_in_client1}_a_{additional_points_per_cluster}_lotoverlap_big_data.csv"

# Writing data to the CSV file
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
