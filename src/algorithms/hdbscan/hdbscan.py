import time

import hdbscan
from mdtraj import Trajectory


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
    start_time = time.time()
    cl = hdbscan.HDBSCAN(min_cluster_size=args.minclustersize, min_samples=args.minsamples)
    cluster_labels = cl.fit_predict(data)
    hdbscan_time = time.time()
    print("--- %s seconds to perform HDBSCAN clustering---" % (hdbscan_time - start_time))

    return cluster_labels
