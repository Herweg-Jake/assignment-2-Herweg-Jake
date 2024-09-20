import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.dataset_file = dataset_file
        self.data = None
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        
    def model_fit(self, n_clusters=3, max_iter=300):
        '''
        Initialize and fit the KMeans model
        '''
        # initialize kmeans
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter)

        self.model.fit(self.data)
        
        return self.model.cluster_centers_

    # def plot_clusters(self):
    #     '''
    #     Plot the data points and the cluster centers
    #     '''
    #     # predict clusters
    #     labels = self.model.predict(self.data)

    #     plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis')
    #     plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1], 
    #                 s=300, c='orange', label='Centroids')
    #     plt.title(f'K-means clustering with {self.model.n_clusters} clusters')
    #     plt.legend()
    #     plt.savefig('kmeans.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    
    classifier = MykmeansClustering(args.dataset_file)
    
    # fit model with optimal clusters
    clusters_centers = classifier.model_fit(n_clusters=3)
    print("Cluster Centers:", clusters_centers)
    
    #classifier.plot_clusters()
