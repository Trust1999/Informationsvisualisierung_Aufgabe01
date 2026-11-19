import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from visualization import Visualization

class Clustering:

    def __init__(self, E):
        self.E = E
        

    def get_kmeans_labels(self, n_clusters=10, n_init=10, random_state=0):
        return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(self.E)

    def get_dbscan_labels(self, eps=9, min_samples=50):
        return DBSCAN(eps=eps,min_samples=min_samples).fit_predict(self.E)

    def get_hdbscan_labels(self, min_cluster_size=50, min_samples=10, cluster_selection_method="eom"):
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method=cluster_selection_method).fit_predict(self.E)

    def get_k_distances(self, n_neighbors=50):
        return NearestNeighbors(n_neighbors=n_neighbors).fit(self.E)

    def get_epsilon_from_k_distances(self, n_neighbors=50, save_plot = True):
        dist, _ = self.get_k_distances(n_neighbors=n_neighbors).kneighbors(self.E) 
        sorted_dist = np.sort(dist[:, -1])
        l = len(sorted_dist)
        
        d = np.zeros(l)
        
        q = np.array([0,sorted_dist[0],0])
        u = np.array([l,np.max(sorted_dist)-np.min(sorted_dist),0])
        
        for i in range(len(sorted_dist)):
            d[i] = np.linalg.norm(np.cross((np.array([i,sorted_dist[i],0])-q),u))/np.linalg.norm(u)
            
        if save_plot:
            Visualization().plot_k_distances(np.arange(0,l,1), sorted_dist, save=True)
        
        return  sorted_dist[np.argmax(d)]