import numpy as np
from algorithms.imwkmeans import clustering
import sklearn.cluster
import matplotlib.pyplot as plt #Vis
import copy

from main.constants import DATA, DATA_DEST

plt.style.use('bmh') #Vis

def cluster(traj, args):
    #Reshape the data
    temp = traj.xyz
    original_data = temp.reshape((traj.xyz.shape[0], traj.xyz.shape[1]*3))
    original_data = original_data.astype('float64')
    temp, traj = [], []

    ## code adapted from Melvin et al.
    np.seterr(all='raise')
    cl = clustering.Clustering()
    if original_data.shape[0] > 10000:
        sample_size = 10000
    else:
        sample_size = None

    original_data = cl.my_math.standardize(original_data)  # Not clear if I should do this

    optimal_p = 2  # From Amorim's experiments

    # Ready to do iMWK-means with explicit rescaling
    # Set an upper bound on k
    [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(original_data, p=optimal_p)
    maxk = max(labels) + 1  # Cluster labels start at 0
    silhouette_averages = np.zeros(maxk-1)

    print("Rescale iMWK trials")
    k_to_try = np.arange(2,maxk+1)
    for k in k_to_try:
        print('Testing k=' + str(k) + ' of ' + str(maxk))
        cl = []
        labels = []
        weights = []
        centroids = []
        cl=clustering.Clustering()
        data = copy.copy(original_data)
        [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, p=optimal_p,k=k)

        # Rescale the data
        for k1 in np.arange(0,max(labels)+1):
            data[labels == k1] = np.multiply(data[labels == k1], np.tile(weights[k1], (np.sum(labels == k1), 1)))
            centroids[k1] = np.multiply(centroids[k1], weights[k1])

        # Apply Euclidean KMeans
        kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=k, n_init=100)
        kmeans_clusters = kmeans_clusterer.fit(data)
        labels = kmeans_clusters.labels_
        centroids = kmeans_clusters.cluster_centers_

        silhouette_averages[k - 2] = silhouette_score(data, labels, sample_size=sample_size)

    optimal_k = k_to_try[np.argmax(silhouette_averages)]
    # Do optimal clustering
    cl=[]
    labels = []
    weights = []
    centroids = []
    cl=clustering.Clustering()
    data = copy.copy(original_data)
    [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, p=optimal_p, k=optimal_k)
    # Rescale the data
    for k1 in np.arange(0,max(labels)+1):
        data[labels == k1] = np.multiply(data[labels == k1], np.tile(weights[k1], (np.sum(labels == k1), 1)))
        centroids[k1] = np.multiply(centroids[k1], weights[k1])

    # Apply Euclidean KMeans
    kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=optimal_k, n_init=100)
    kmeans_clusters = kmeans_clusterer.fit(data)
    labels = kmeans_clusters.labels_
    centroids = kmeans_clusters.cluster_centers_
    silhouette_score = silhouette_score(data, labels, sample_size=sample_size)

    np.savetxt(DATA + DATA_DEST + args.destination + 'RescalediMWK_labels.txt', labels, fmt='%i')
    with open(DATA + DATA_DEST + args.destination + 'imwk_silhouette_score.txt', 'w') as f:
        f.write("silhouette score is {0} \n with p of {1}\n".format(silhouette_score, optimal_p))

    plt.figure()
    plt.scatter(np.arange(labels.shape[0]), labels, marker='+')
    plt.xlabel('Frame')
    plt.ylabel('Cluster')
    plt.title('iMWK-means with Explicit Rescaling and Kmeans')
    plt.savefig(DATA + DATA_DEST + args.destination + 'RescalediMWK_timeseries.png')
    if args.visualise:
        plt.show()
    plt.clf()
