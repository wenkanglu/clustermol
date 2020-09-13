import matplotlib.pyplot as plt
import os
import hdbscan
from mdtraj import Trajectory


plt.style.use('bmh')

directory = os.getcwd()


def cluster(input, args):
    data = None
    if isinstance(input, Trajectory):
        # Reshape the data
        temp = input.xyz
        data = temp.reshape((input.xyz.shape[0], input.xyz.shape[1]*3))
        data = data.astype('float64')

    else:
        data = input

    print("Performing HDBSCAN clustering.")
    cl = hdbscan.HDBSCAN(min_cluster_size=args.minclustersize, min_samples=args.minsamples)
    cluster_labels = cl.fit_predict(data)

    return cluster_labels
