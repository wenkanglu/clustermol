import numpy as np
import sklearn.cluster
import sklearn.metrics #Val
import matplotlib.pyplot as plt #Vis
import hdbscan
from mdtraj import Trajectory

from main.constants import DATA, DATA_DEST

plt.style.use('bmh') #Vis


def cluster(traj, args):
    if isinstance(traj, Trajectory):
        #Reshape the data
        temp = traj.xyz
        data = temp.reshape((traj.xyz.shape[0], traj.xyz.shape[1]*3))
        data = data.astype('float64')
        temp, traj = [], []

    else:
        data = traj

    print("Performing HDBSCAN clustering.")
    ## code adapted from Melvin et al.
    cl = hdbscan.HDBSCAN(min_cluster_size=int(args.minclustersize), min_samples=int(args.minsamples)) #min cluster size -> parameter?
    cluster_labels = cl.fit_predict(data)
    if data.shape[0] > 10000:
        sample_size = 10000
    else:
        sample_size = None
    raw_score = sklearn.metrics.silhouette_score(data, cluster_labels, sample_size=sample_size)

    np.savetxt(DATA + DATA_DEST + args.destination + 'hdbscan_labels.txt', cluster_labels, fmt='%i')
    with open(DATA + DATA_DEST + args.destination + 'hdb_silhouette_score.txt', 'w') as f:
        f.write("silhouette score is {0} \n".format(raw_score))

    plt.figure()
    plt.scatter(np.arange(cluster_labels.shape[0]), cluster_labels, marker = '+')
    plt.xlabel('Frame')
    plt.ylabel('Cluster')
    plt.title('HDBSCAN')
    plt.savefig(DATA + DATA_DEST + args.destination + 'hdbscan_timeseries.png')
    if(args.visualise):
        plt.show()
    plt.clf()

    #print("Noise: ", label_counts(cluster_labels)[-1])
