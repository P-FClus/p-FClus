import scipy.sparse as sps
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances as sparse_cdist
import csv
from sklearn.utils.extmath import randomized_svd
from IPython.utils.text import string
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import random

def pFClus(global_centers, local_centers, X, k, max_iters=10):
    """
    Parameters:
    -global_centers: Global cluster centroids for a client.
    -local_centers: Local cluster centroids for a client.
    - X: Input data as a numpy array with shape (n_samples, n_features).
    - k: Number of clusters.
    - max_iters: Maximum number of iterations.
    - learning_rate: Learning rate for gradient descent.
    
    Returns:
    - centroids: Final cluster centroids.
    - labels: centroid assignments for each data point.
    """
    n_samples, n_features = X.shape

    # Initialize cluster centroids for the client
    centroids = global_centers
    count = np.zeros(total_classes)
    fine_tune = 0.01
    
    for _ in range(max_iters):
        for i in range(n_samples):
            # select a data point
            data_point = X[i]

            # Calculate the distance to each local centroid
            local_distances = np.linalg.norm(local_centers - data_point, axis=1)
            
            # Find the index of the closest local centroid
            local_closest_centroid_index = np.argmin(local_distances)
            
            # Calculate the distance to each global centroid
            global_distances = np.linalg.norm(centroids - data_point, axis=1)
            
            # Find the index of the closest global centroid
            global_closest_centroid_index = np.argmin(global_distances)
            
            #updating the corresponding global centroid
            count[global_closest_centroid_index]+=1
            learning_rate = 1/count[global_closest_centroid_index]
            centroids[global_closest_centroid_index] -= learning_rate * ((centroids[global_closest_centroid_index] - data_point )+2*fine_tune*(centroids[global_closest_centroid_index] - local_centers[local_closest_centroid_index]))

    centroids = np.array(centroids)
    distances = np.linalg.norm(centroids[:, np.newaxis] - X, axis=2)
    labels = np.argmin(distances, axis=0)
    return centroids,labels

def sgd_scipy(X,k):
    """
    Parameters:
    - X: Input data as a numpy array with shape (n_samples, n_features).
    - k: Number of clusters.
    
    Returns:
    - centroids: Final cluster centroids.
    - labels: centroid assignments for each data point.
    """
    #applying Kmeans to get the cluster centers and labels
    kmeans1 = KMeans(n_clusters=k)
    kmeans1.fit(X)
    sgd_centers = kmeans1.cluster_centers_
    sgd_labels = kmeans1.labels_
    return sgd_centers,sgd_labels

#obtain the data in form of numpy array
adult_data = pd.read_csv('adult.csv', header=None) #reading the data
data_arrays = adult_data.values.tolist() #converting the data to list
data_arrays = np.array(data_arrays) #converting the data to numpy array

data = []
#split the data_arrays into 100 clients (code for the same is provided in //)

seeds = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]

clustering_cost_per_run = []
stdev_per_run = []
max_cost_per_run = []

total_classes = 10 #number of clusters
total_devices = len(data) #number of clients
num_clusters = total_classes 
k = total_classes

for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)

    clust_cost = []

    #calculating average clustering cost per client
    def cost(centroids,labels,data):
        sum = 0
        for i in range(data.shape[0]):
            center_assigned = centroids[labels[i]]
            distance = np.linalg.norm(data[i]-center_assigned)
            sum+=distance
        sum/=data.shape[0]
        return sum

    centroids = []
    labels = []

    #calculating the cluster centers and labels for each client initially
    for i in range(total_devices):
        centers, labels_found = sgd_scipy(data[i], min(k,len(data[i])))
        centers = np.array(centers)
        centroids.append(centers)
        labels.append(labels_found)

    centroids_center = []

    for i in range(total_devices):
        for j in range(len(centroids[i])):
            centroids_center.append(centroids[i][j])

    centroids_center = np.array(centroids_center)

    kmeans1 = KMeans(n_clusters=total_classes)
    kmeans1.fit(centroids_center)

    # Finding global centers by the server
    final_centers = kmeans1.cluster_centers_

    ans_pFClus = []
    labels_pFClus = []

    #finding final personalised centroids and labels for each client
    for i in range(total_devices):
        pFClus_centroids,pFClus_labels = pFClus(final_centers,centroids[i],data[i],k = min(total_classes,len(data[i])))
        ans_pFClus.append(pFClus_centroids)
        labels_pFClus.append(pFClus_labels)

    ans_pFClus = np.array(ans_pFClus)

    #calculating the average clustering cost for each client
    for i in range(total_devices):
        cost_of_pFClus = cost(ans_pFClus[i],labels_pFClus[i],data[i])
        clust_cost.append(cost_of_pFClus)

    mean_pFClus = sum(clust_cost)/total_devices
    clustering_cost_per_run.append(mean_pFClus)
    max_pFClus = max(clust_cost)
    max_cost_per_run.append(max_pFClus)

    #std_dev
    variance_pFClus = 0
    for i in range (total_devices):
        variance_pFClus += (clust_cost[i] - mean_pFClus)**2
        
    variance_pFClus /= total_devices

    std_dev_pFClus = variance_pFClus**0.5

    stdev_per_run.append(std_dev_pFClus)

# mean and std dev of mean
num_runs = len(seeds)
avg_mean_pFClus = sum(clustering_cost_per_run)/num_runs

avg_mean_variance_pFClus = 0

for i in range (num_runs):
    avg_mean_variance_pFClus += (clustering_cost_per_run[i] - avg_mean_pFClus)**2

avg_mean_variance_pFClus /= num_runs

stdev_mean_pFClus = avg_mean_variance_pFClus**0.5

# mean and std dev of std dev
avg_stdev_pFClus = sum(stdev_per_run)/num_runs

avg_stdev_variance_pFClus = 0
for i in range (num_runs):
    avg_stdev_variance_pFClus += (stdev_per_run[i] - avg_stdev_pFClus)**2

avg_stdev_variance_pFClus /= num_runs

stdev_std_dev_pFClus = avg_stdev_variance_pFClus**0.5

# mean and std dev of max_pFClus
avg_max_pFClus = sum(max_cost_per_run)/num_runs

avg_max_variance_pFClus = 0
for i in range (num_runs):
    avg_max_variance_pFClus += (max_cost_per_run[i] - avg_max_pFClus)**2

avg_max_variance_pFClus /= num_runs

stdev_max_pFClus = avg_max_variance_pFClus**0.5

info_pFClus=[]
info_pFClus.append(avg_mean_pFClus)
info_pFClus.append(stdev_mean_pFClus)
info_pFClus.append(avg_stdev_pFClus)
info_pFClus.append(stdev_std_dev_pFClus)
info_pFClus.append(avg_max_pFClus)
info_pFClus.append(stdev_max_pFClus)

info = []
info.append(info_pFClus)

# Define the filename for the CSV file
csv_filename = 'result_pFClus.csv'

# Save the array to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(info)