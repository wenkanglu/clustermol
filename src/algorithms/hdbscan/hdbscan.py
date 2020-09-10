import numpy as np
from processing import post_proc
import sklearn.cluster
import matplotlib.pyplot as plt #Vis
import hdbscan
from mdtraj import Trajectory

from main.constants import DATA, DATA_DEST

plt.style.use('bmh')  # Vis


def cluster(input, args):
    data = None
    if isinstance(input, Trajectory):
        #Reshape the data
        temp = input.xyz
        data = temp.reshape((input.xyz.shape[0], input.xyz.shape[1]*3))
        data = data.astype('float64')
        temp, input = [], []

    else:
        data = input

    print("Performing HDBSCAN clustering.")
    cl = hdbscan.HDBSCAN(min_cluster_size=int(args.minclustersize), min_samples=int(args.minsamples))
    cluster_labels = cl.fit_predict(data)

    np.savetxt(DATA + DATA_DEST + args.destination + 'hdbscan_labels.txt', cluster_labels, fmt='%i')
    post_proc.label_counts(cluster_labels, args.destination)
    if args.validate:
        post_proc.calculate_CVI(args.validate, data, cluster_labels, args.destination)

        plt.figure()
        plt.scatter(np.arange(cluster_labels.shape[0]), cluster_labels, marker='+')
        plt.xlabel('Frame')
        plt.ylabel('Cluster')
        plt.title('HDBSCAN')
        plt.savefig(DATA + DATA_DEST + args.destination + 'hdbscan_timeseries.png')

    if args.visualise:
        plt.show()
        plt.clf()
        clustered = (cluster_labels >= 0)
        plt.scatter(data[clustered, 0],
                    data[clustered, 1],
                    c=cluster_labels[clustered],
                    s=1,
                    cmap='Set1')
        plt.legend(cluster_labels[clustered])
        plt.savefig(DATA + DATA_DEST + args.destination + 'hdbscan_scatter.png')
        plt.show()

    return cluster_labels
