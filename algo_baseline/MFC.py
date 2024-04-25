import numpy as np
import scipy
from sklearn.metrics import pairwise_distances as sparse_cdist
from sklearn.utils.extmath import randomized_svd
import csv
import scipy.sparse as sps
from IPython.utils.text import string
import io
import re
from collections import defaultdict
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from sklearn.cluster import KMeans
import pandas as pd
import random

class MFC:
    def __init__(self):
        pass

    def distance_to_set(self, A, S, sparse=False):
        '''
        S is a list of points. Distance to set is the minimum distance of $x$ to
        points in $S$. In this case, this is computed for each row of $A$.  Note
        that this works with sparse matrices (sparse=True)
        Returns a single array of length len(A) containing corresponding distances.
        '''
        n, d = A.shape
        assert S.ndim == 2
        assert S.shape[1] == d, S.shape[1]
        assert A.shape[1] == d
        assert A.ndim == 2
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
        else:
            pd = sparse_cdist(A, S)
        assert np.allclose(pd.shape, [A.shape[0], len(S)])
        dx = np.min(pd, axis=1)
        assert len(dx) == A.shape[0]
        assert dx.ndim == 1
        return dx


    def get_clustering(self, A, centers, sparse=False):
        '''
        Returns a list of integers of length len(A). Each integer is an index which
        tells us the cluster A[i] belongs to. A[i] is assigned to the closest
        center.
        '''
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
        else:
            pd = sparse_cdist(A, centers)
        assert np.allclose(pd.shape, [A.shape[0], len(centers)])
        indices = np.argmin(pd, axis=1)
        assert len(indices) == A.shape[0]
        return np.array(indices)


    def kmeans_cost(self, A, centers, sparse=False, remean=False):
        '''
        Computes the k means cost of rows of $A$ when assigned to the nearest
        centers in `centers`.
        remean: If remean is set to True, then the kmeans cost is computed with
        respect to the actual means of the clusters and not necessarily the centers
        provided in centers argument (which might not be actual mean of the
        clustering assignment).
        '''
        clustering = self.get_clustering(A, centers, sparse=sparse)
        cost = 0
        if remean is True:
            # We recompute mean based on assignment.
            centers2 = []
            for clusterid in np.unique(clustering):
                points = A[clustering == clusterid]
                centers2.append(np.mean(points, axis=0))
            centers = np.array(centers2)
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            dist = self.distance_to_set(points, centers, sparse=sparse)
            cost += np.mean(dist ** 2)
        return cost

    def kmeans_pp(self, A, k, weighted=True, sparse=False, verbose=False):
        '''
        Returns $k$ initial centers based on the k-means++ initialization scheme.
        With weighted set to True, we have the standard algorithm. When weighted is
        set to False, instead of picking points based on the D^2 distribution, we
        pick the farthest point from the set (careful deterministic version --
        affected by outlier points). Note that this is not deterministic.
        A: nxd data matrix (sparse or dense). 
        k: is the number of clusters.
        Returns a (k x d) dense matrix.
        K-means ++
        ----------
        1. Choose one center uniformly at random among the data points.
        2. For each data point x, compute D(x), the distance between x and
            the nearest center that has already been chosen.
        3. Choose one new data point at random as a new center, using a
            weighted probability distribution where a point x is chosen with
            probability proportional to D(x)2.
        4. Repeat Steps 2 and 3 until k centers have been chosen.
        '''
        n, d = A.shape
        if n <= k:
            if sparse:
                A = A.toarray()
            return np.aray(A)
        index = np.random.choice(n)
        if sparse is True:
            B = np.squeeze(A[index].toarray())
            assert len(B) == d
            inits = [B]
        else:
            inits = [A[index]]
        indices = [index]
        t = [x for x in range(A.shape[0])]
        distance_matrix = self.distance_to_set(A, np.array(inits), sparse=sparse)
        distance_matrix = np.expand_dims(distance_matrix, axis=1)
        while len(inits) < k:
            if verbose:
                print('\rCenter: %3d/%4d' % (len(inits) + 1, k), end='')
            # Instead of using distance to set we can compute this incrementally.
            dx = np.min(distance_matrix, axis=1)
            assert dx.ndim == 1
            assert len(dx) == n
            if np.sum(dx**2)==0:
                continue
            else:
                dx = dx**2/np.sum(dx**2)
            if weighted:
                choice = np.random.choice(t, 1, p=dx)[0]
            else:
                choice = np.argmax(dx)
            if choice in indices:
                continue
            if sparse:
                B = np.squeeze(A[choice].toarray())
                assert len(B) == d
            else:
                B = A[choice]
            inits.append(B)
            indices.append(choice)
            last_center = np.expand_dims(B, axis=0)
            assert last_center.ndim == 2
            assert last_center.shape[0] == 1
            assert last_center.shape[1] == d
            dx = self.distance_to_set(A, last_center, sparse=sparse)
            assert dx.ndim == 1
            assert len(dx) == n
            dx = np.expand_dims(dx, axis=1)
            a = [distance_matrix, dx]
            distance_matrix = np.concatenate(a, axis=1)
        if verbose:
            print()
        return np.array(inits)


    def awasthisheffet(self, A, k, useSKLearn=True, sparse=False):
        '''
        The implementation here uses kmeans++ (i.e. probabilistic) to get initial centers 
        (\nu in the paper) instead of using a 10-approx algorithm.
        1. Project onto $k$ dimensional space.
        2. Use $k$-means++ to initialize.
        3. Use 1:3 distance split to improve initialization.
        4. Run Lloyd steps and return final solution.
        Returns a sklearn.cluster.Kmeans object with the clustering information and
        the list $S_r$.
        '''
        assert A.ndim == 2
        n = A.shape[0]
        d = A.shape[1]
        # If we don't have $k$ points then return the matrix as its the best $k$
        # partition trivially.
        if n <= k:
            if sparse:
                A = np.array(A.toarray())
            return A, None
        # This works with sparse and dense matrices. Returns dense always.
        # Randomized though so average.
        U, Sigma, V = randomized_svd(A, n_components=k, random_state=None)
        # Columns of $V$ are eigen vectors
        V = V.T[:, :k]
        # Sparse and dense compatible. A_hat is always dense.
        A_hat = A.dot(V)
        inits = self.kmeans_pp(A_hat, k, sparse=False)
        # Run STEP 2, modified Lloyd. We have vectorized it for speed up.
        if sparse is False:
            pd = scipy.spatial.distance.cdist(inits, A_hat)
        else:
            pd = sparse_cdist(inits, A_hat)
        Sr_list = []
        for r in range(k):
            th = 3 * pd[r, :]
            remaining_dist = pd[np.arange(k) != r]
            assert np.allclose(remaining_dist.shape, [k- 1, n])
            indicator = (remaining_dist - th) < 0
            indicator = np.sum(indicator.astype(int), axis=0)
            assert len(indicator) == n
            # places where indicator is 0 is our set
            Sr = [i for i in range(len(indicator)) if indicator[i] == 0]
            assert len(Sr) >= 0
            Sr_list.append(Sr)
        # We don't mind lloyd_init being dense. Its only k x d.
        lloyd_init = np.array([np.mean(A_hat[Sr], axis=0) for Sr in Sr_list])

        ''' COMMENTED THIS '''
        # assert np.allclose(lloyd_init.shape, [k, k])
        # Project back to d dimensional space
        lloyd_init = np.matmul(lloyd_init, V.T)
        assert np.allclose(lloyd_init.shape, [k, d])
        # Run Lloyd's method
        if useSKLearn:
            # Works with sparse matrices as well.
            kmeans = KMeans(n_clusters=k, init=lloyd_init)
            kmeans.fit(A)
            ret = (kmeans.cluster_centers_, kmeans)
        else:
            raise NotImplementedError()
        # We use the GPU version from torch:with
        return ret


    def kfed(self, x_dev, dev_k, k, useSKLearn=True, sparse=False):
        dev_k = k
        '''
        The full decentralized algorithm.
        Warning: Synchronous version, no parallelization across devices. Since the
        sklearn k means routine is itself parallel. 
        x_dev: [Number of devices, data length, data dimension]
        dev_k: Device k (int). The value $k'$ in the paper. Number of clusters
            per device. We use constant for all devices.
        https://further-reading.net/2017/01/quick-tutorial-python-multiprocessing/
        Returns: Local estimators (local centers), central-centers
        '''
        def cleaup_max(local_estimators, k, dev_k, useSKLearn=True, sparse=False):
            '''
            Central cleanup phase based on the max-from-set rule.
            
            Switch to either percentile rule or probabilistic (kmeans++) rule in
            case of outlier points.
            '''
            assert local_estimators.ndim == 2
            # The first dev_k points definitely in different target clusters.
            init_centers = local_estimators[:1, :]
            remaining_data = local_estimators[1:, :]
            # For the remaining initialization, use max rule.
            while len(init_centers) < k:
                distances = self.distance_to_set(remaining_data, np.array(init_centers),
                                            sparse=sparse)
                candidate_index = np.argmax(distances)
                candidate = remaining_data[candidate_index:candidate_index+1, :]
                # Combine with init_centers
                init_centers = np.append(init_centers, candidate, axis=0)
                # Remove from remaining_data
                remaining_data = np.delete(remaining_data, candidate_index, axis=0)

            assert len(init_centers) == k
            # Perform final clustering.
            if useSKLearn:
                # Works with sparse matrices as well.
                kmeans = KMeans(n_clusters=k, init=init_centers)
                kmeans.fit(local_estimators)
                ret = (kmeans.cluster_centers_, kmeans)
            else:
                raise NotImplementedError("This is not implemented/tested")
            return ret

        num_dev = len(x_dev)
        msg = "Not enough devices "
        msg += "(num_dev=%d, dev_k=%d, k=%d)" % (num_dev, dev_k, k)
        assert dev_k * num_dev >= k, msg
        # Run local $k$-means
        local_clusters = []
        for dev in x_dev:
            cluster_centers, _ = self.awasthisheffet(dev, dev_k, useSKLearn=useSKLearn,
                                                sparse=sparse)
            local_clusters.append(cluster_centers)
        
        # This is alwasys dense.
        local_estimates = np.concatenate(local_clusters, axis=0)
        msg = "Not enough estimators. "
        msg += "Estimator matrix size: " + str(local_estimates.shape) + ", while "
        msg += "k = %d" % k
        assert local_estimates.shape[0] > k, msg
        # Local estimators are dense
        centers, kmeansobj = cleaup_max(local_estimates, k, dev_k,
                                        useSKLearn=useSKLearn, sparse=False)
        return local_estimates, centers
     
     #added by us to calculate the clustering cost of MFC
    def clust_cost(self,data,centers ,sparse=False, remean=False):
        
        costs = []
        for device in data:
            clustering = self.get_clustering(device, centers, sparse=sparse)
            cost = 0
            if remean is True:
                # We recompute mean based on assignment.
                centers2 = []
                for clusterid in np.unique(clustering):
                    points = device[clustering == clusterid]
                    centers2.append(np.mean(points, axis=0))
                centers = np.array(centers2)
            
            for clusterid in np.unique(clustering):
                points = device[clustering == clusterid]
                dist = self.distance_to_set(points, centers, sparse=sparse)
                cost +=sum(dist)
            costs.append(cost/len(device))
        return costs

class COST:
    def distance_to_set(self, A, S, sparse=False):
        '''
        S is a list of points. Distance to set is the minimum distance of $x$ to
        points in $S$. In this case, this is computed for each row of $A$.  Note
        that this works with sparse matrices (sparse=True)
        Returns a single array of length len(A) containing corresponding distances.
        '''
        n, d = A.shape
        assert S.ndim == 2
        assert S.shape[1] == d, S.shape[1]
        assert A.shape[1] == d
        assert A.ndim == 2
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
        else:
            pd = sparse_cdist(A, S)
        assert np.allclose(pd.shape, [A.shape[0], len(S)])
        dx = np.min(pd, axis=1)
        assert len(dx) == A.shape[0]
        assert dx.ndim == 1
        return dx

    def get_clustering(self, A, centers, sparse=False):
        '''
        Returns a list of integers of length len(A). Each integer is an index which
        tells us the cluster A[i] belongs to. A[i] is assigned to the closest
        center.
        '''
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
        else:
            pd = sparse_cdist(A, centers)
        assert np.allclose(pd.shape, [A.shape[0], len(centers)])
        indices = np.argmin(pd, axis=1)
        assert len(indices) == A.shape[0]
        return np.array(indices)

    def kmeans_cost(self, A, centers, sparse=False, remean=False):
        '''
        Computes the k means cost of rows of $A$ when assigned to the nearest
        centers in `centers`.
        remean: If remean is set to True, then the kmeans cost is computed with
        respect to the actual means of the clusters and not necessarily the centers
        provided in centers argument (which might not be actual mean of the
        clustering assignment).
        '''
        clustering = self.get_clustering(A, centers, sparse=sparse)
        cost = 0

        centers2 = []
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            centers2.append(np.mean(points, axis=0))
        centers2 = np.array(centers2)
        dic = defaultdict(lambda: 0)
        arr = []
        # print(list(np.unique(clustering)))

        max_cost = 0
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            dist = self.distance_to_set(points, centers, sparse=sparse)
            
            dist2 = self.distance_to_set(points, centers2, sparse=sparse)

            ''' CHANGED '''
            cost += np.mean(dist ** 2)
            # cost += sum(dist)
            
            dic[tuple(centers[clusterid])] += np.mean(dist**2)
            arr.append([np.mean(dist2**2),points])
        arr.sort(key=lambda x: x[0], reverse=True)
        return arr

def cleaup_max(local_estimators, weights, k, dev_k, useSKLearn=True, sparse=False):
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(local_estimators, sample_weight = weights)
    ret = (kmeans.cluster_centers_, kmeans)
    return ret

def get_optimal_k(x):
    # create an empty list to store the Within-Cluster-Sum-of-Squares (WCSS) values
    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    return sil.index(max(sil)) + 2

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

    algos = [MFC]

    for algo_class in algos:
        
        algo = algo_class()
        avg_cost = [0]*11           
        a,centers = algo.kfed(data, total_classes, total_classes) #locally centers for kfed use this only
        calc = COST()
        centers_cost = defaultdict(lambda: [])
        shots = 10
        combined_data = []
        
        for i in data:
            combined_data.extend(i)
        combined_data = np.array(combined_data) #combined data in 2d form
        cost = algo.kmeans_cost(combined_data, centers)
        avg_cost[0] += cost

        for shot in range (shots):
            new_pts = []
            weights = []
            for device_id in range (len(data)):
                device = data[device_id]
                arr = calc.kmeans_cost(device, centers)
                for cost, points in arr[:1]:
                    # if len(points)<len(data[device_id])//(2*K):
                    #     continue
                    dim = len(points[0])
                    filtered_pts = [[] for i in range (dim)]
                    for point in points:
                        for axis in range (dim):
                            filtered_pts[axis].append(point[axis])
                    new_pt = []
                    for axis in range (dim):
                        new_pt.append(np.mean(filtered_pts[axis]))
                    new_pts.append(new_pt)
                    weights.append(len(points))
                    break
            
            max_weight = max(weights)
            new_pts.extend(centers)
            weights.extend([max_weight]*len(centers))
            
            new_pts = np.array(new_pts)
            weights = np.array(weights)
            centers, _ = cleaup_max(new_pts, weights, total_classes, 1)
            cost = algo.kmeans_cost(combined_data, centers) #yaha lagao mfc and mfc_h
            avg_cost[shot+1]+=cost
    
            MFC_clust_cost = algo.clust_cost(data,centers) 

    mean_mfc = sum(clust_cost)/total_devices
    clustering_cost_per_run.append(mean_mfc)
    max_mfc = max(clust_cost)
    max_cost_per_run.append(max_mfc)

    #std_dev
    variance_mfc = 0
    for i in range (total_devices):
        variance_mfc += (clust_cost[i] - mean_mfc)**2
        
    variance_mfc /= total_devices

    std_dev_mfc = variance_mfc**0.5

    stdev_per_run.append(std_dev_mfc)


# mean and std dev of mean
num_runs = len(seeds)
avg_mean_mfc = sum(clustering_cost_per_run)/num_runs

avg_mean_variance_mfc = 0

for i in range (num_runs):
    avg_mean_variance_mfc += (clustering_cost_per_run[i] - avg_mean_mfc)**2

avg_mean_variance_mfc /= num_runs

stdev_mean_mfc = avg_mean_variance_mfc**0.5

# mean and std dev of std dev
avg_stdev_mfc = sum(stdev_per_run)/num_runs

avg_stdev_variance_mfc = 0
for i in range (num_runs):
    avg_stdev_variance_mfc += (stdev_per_run[i] - avg_stdev_mfc)**2

avg_stdev_variance_mfc /= num_runs

stdev_std_dev_mfc = avg_stdev_variance_mfc**0.5

# mean and std dev of max_mfc
avg_max_mfc = sum(max_cost_per_run)/num_runs

avg_max_variance_mfc = 0
for i in range (num_runs):
    avg_max_variance_mfc += (max_cost_per_run[i] - avg_max_mfc)**2

avg_max_variance_mfc /= num_runs

stdev_max_mfc = avg_max_variance_mfc**0.5

info_mfc=[]
info_mfc.append(avg_mean_mfc)
info_mfc.append(stdev_mean_mfc)
info_mfc.append(avg_stdev_mfc)
info_mfc.append(stdev_std_dev_mfc)
info_mfc.append(avg_max_mfc)
info_mfc.append(stdev_max_mfc)

info = []
info.append(info_mfc)

# Define the filename for the CSV file
csv_filename = 'result_mfc.csv'

# Save the array to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(info)