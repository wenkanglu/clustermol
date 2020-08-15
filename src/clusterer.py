import mdtraj as md
import numpy as np
import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, 'src/deAmorim')
import clustering
import sklearn.metrics
import sklearn.cluster
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
import scipy
import copy

def cluster_imwkmeans(args):
    print("Loading trajectory from file...") # eventually move this to cmd & add selection options
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+"/data/data_src")
    t = md.load(args.source)
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    ## code adapted from Melvin et al.
    temp = t.xyz
    frames = t.xyz.shape[0]
    atoms = t.xyz.shape[1]
    original_data = temp.reshape((frames,atoms*3))
    original_data = original_data.astype('float64')
    temp = []
    t = []

    np.seterr(all='raise')
    cl = clustering.Clustering()
    if frames > 10000:
        sample_size = 10000
    else:
        sample_size = None

    original_data = cl.my_math.standardize(original_data) #Not clear if I should do this

    optimal_p=2 #From Amorim's experiments

    # Ready to do iMWK-means with explicit rescaling
    # Set an upper bound on k
    [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(original_data, p=optimal_p)
    maxk = max(labels) + 1 #Cluster labels start at 0
    silhouette_averages = np.zeros(maxk-1)

    print("Rescale iMWK trials")
    k_to_try = np.arange(2,maxk+1)
    for k in k_to_try:
        print('Testing k=' + str(k) + ' of ' +  str(maxk))
        cl=[]
        labels = []
        weights = []
        centroids = []
        cl=clustering.Clustering()
        data = copy.copy(original_data)
        [labels, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, p=optimal_p,k=k)

        # Rescale the data
        for k1 in np.arange(0,max(labels)+1):
            data[labels==k1] = np.multiply(data[labels==k1],np.tile(weights[k1], (np.sum(labels==k1),1)))
            centroids[k1] = np.multiply(centroids[k1], weights[k1])

        # Apply Euclidean KMeans
        kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=k, n_init=100)
        kmeans_clusters = kmeans_clusterer.fit(data)
        labels = kmeans_clusters.labels_
        centroids = kmeans_clusters.cluster_centers_

        silhouette_averages[k - 2] = sklearn.metrics.silhouette_score(data, labels, sample_size=sample_size)

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
        data[labels==k1] = np.multiply(data[labels==k1],np.tile(weights[k1], (np.sum(labels==k1),1)))
        centroids[k1] = np.multiply(centroids[k1], weights[k1])

    # Apply Euclidean KMeans
    kmeans_clusterer = sklearn.cluster.KMeans(n_clusters=optimal_k, n_init=100)
    kmeans_clusters = kmeans_clusterer.fit(data)
    labels = kmeans_clusters.labels_
    centroids = kmeans_clusters.cluster_centers_
    silhouette_score = sklearn.metrics.silhouette_score(data, labels, sample_size=sample_size)

    np.savetxt('data/data_dest/' + args.destination + 'RescalediMWK_labels.txt', labels, fmt='%i')
    with open ('data/data_dest/' + args.destination + 'silhouette_score.txt', 'w') as f:
        f.write("silhouette score is {0} \n with p of {1}\n".format(silhouette_score,optimal_p))

    #Figures
    plt.figure()
    plt.scatter(np.arange(frames), labels, marker='+')
    plt.xlabel('Frame')
    plt.ylabel('Cluster')
    plt.title('iMWK-means with Explicit Rescaling and Kmeans')
    plt.savefig('data/data_dest/' + args.destination + 'RescalediMWK_timeseries.png')
    if(args.visualise):
        plt.show()
    plt.clf()
