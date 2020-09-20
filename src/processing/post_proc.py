import numpy as np
import matplotlib.pyplot as plot
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from mdtraj import Trajectory
from itertools import cycle, islice

from main.constants import SILHOUETTE, DAVIESBOULDIN, CALINSKIHARABASZ, DATA_DEST, DATA_SRC, DATA, IRIS, WINE, \
    BREASTCANCER, DIGITS, BLOBS, VBLOBS, NOISE


def label_counts(labels, test=None, selection=None, type=None, dest=None):
    actual_labels = None
    artificial = [BLOBS, VBLOBS, NOISE]
    if test == IRIS:
        actual_labels = load_iris().target
    elif test == WINE:
        actual_labels = load_wine().target
    elif test == DIGITS:
        actual_labels = load_digits().target
    elif test == BREASTCANCER:
        actual_labels = load_breast_cancer().target

    unique, counts = np.unique(labels, return_counts=True)
    d = dict(zip(unique, counts))
    if dest and type:
        with open(DATA + DATA_DEST + dest + "/results-%s.txt" % type, 'w') as f:
            f.write("total frames: {0}\n".format(len(labels)))
            if selection:
                f.write("selection statement used: %s\n" % selection)
            f.write("cluster label: frame count\n")
            for k in d.keys():
                f.write("{0}: {1}\n".format(k, d[k]))
            if test is not None and test not in artificial:
                f.write("Accuracy score: %s\n" % str(accuracy_score(labels, actual_labels)))
    return d


def calculate_CVI(indices, input_data, labels, dest, type, ignore_noise=False):
    data = None
    if isinstance(input_data, Trajectory):
        # Reshape the data
        temp = input_data.xyz
        data = temp.reshape((input_data.xyz.shape[0], input_data.xyz.shape[1]*3))
        data = data.astype('float64')
        temp, input_data = [], []

    else:
        data = input_data

    if ignore_noise and -1 in labels:
        to_rem = []
        for l in range(len(labels)):
            if labels[l] == -1:
                to_rem.append(l)
        labels = np.delete(labels, to_rem)
        data = np.delete(data, to_rem, 0)

    with open(DATA + DATA_DEST + dest + "/results-%s.txt" % type, 'a') as f:
        if ignore_noise:
            f.write("--CVI results with noise ignored--\n")
        else:
            f.write("--CVI results--\n")
        if SILHOUETTE in indices:
            sample_size = 10000 if data.shape[0] > 10000 else None
            f.write("Silhouette score is {0}\n".format(silhouette_score(data, labels, sample_size=sample_size)))
        if DAVIESBOULDIN in indices:
            f.write("Davies-Bouldin score is {0}\n".format(davies_bouldin_score(data, labels)))
        if CALINSKIHARABASZ in indices:
            f.write("Calinski and Harabasz score is {0}\n".format(calinski_harabasz_score(data, labels)))


def cophenetic(linkage_matrix, rmsd_matrix):
    '''
    DESCRIPTION
    Computes Cophenetic distance given linkage and rmsd_matrix.

    Arguments:
        linkage_matrix (numpy.ndarray): cluster linkage.
        rmsd_matrix (numpy.ndarray): pairwise distance matrix.
    '''
    reduced_distances = squareform(rmsd_matrix, checks=True)
    c, coph_dists = cophenet(linkage_matrix, pdist(reduced_distances))
    print(">>> Cophenetic Distance: %s" % c)


def plotTestData(data, labels, dest):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(labels) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plot.figure(figsize=(8, 6))
    plot.title('Test data cluster scatterplot')
    plot.xlabel('x coordinates')
    plot.ylabel('y coordinates')
    plot.scatter(data[:, 0], data[:, 1], c =labels, cmap =plot.cm.Set1)
    plot.savefig(DATA + DATA_DEST + dest + "/test-scatter.png", dpi=300)
    plot.close()


def save_largest_clusters(n, traj, labels, dest, type):
    c = label_counts(labels)
    n_labels = []
    c.pop(-1, None) #ignore "noise" cluster
    for i in range(min(n, len(c))):
        max_key = max(c, key=c.get)
        n_labels.append(max_key)
        c.pop(max_key)
    trajectories = [None]*len(n_labels)
    il = 0
    for l in labels:
        if l in n_labels:
            j = n_labels.index(l)
            if trajectories[j]:
                trajectories[j] = trajectories[j].join(traj[il])
            else:
                trajectories[j] = traj[il]
        il += 1
    for k in n_labels:
        ik = n_labels.index(k)
        trajectories[ik].save_pdb(DATA + DATA_DEST + dest + "/%s-cluster%d" %(type, k) + ".pdb")


def save_without_noise(traj, labels, save = False):
    noiseless = None
    start = 0
    il = 0
    for l in labels:
        if l != -1:
            if noiseless:
                noiseless = noiseless.join(traj[start:il-1])
            else:
                noiseless = traj[start:il-1]
            start = il+1
        il += 1
    if noiseless:
        noiseless = noiseless.join(traj[start:il])
    else:
        noiseless = traj[start:il]
    if save:
        noiseless.save_pdb(DATA + DATA_SRC + "nonoise.pdb")
    return noiseless


def saveClusters(labels, dest, type):
    '''
    DESCRIPTION
    Save cluster indexes to text file

    Arguments:
        clusters_arr (numpy.ndarray): Cluster indexes per frame.
        dest (str): destination to save cluster labels.
        type (str): type of algorithm implementation.
    '''
    np.savetxt(DATA + DATA_DEST + dest + "/clusters-%s.txt" %type, labels, fmt='%i')


def scatterplot_cluster(labels, dest, type, preprocess, test, visualise):
    '''
    DESCRIPTION
    Produce cluster scatter plot of frames

    Arguments:
        labels (numpy.ndarray): cluster indexes per frame.
        no_frames (int): number of frames
        dest (str): destination to save scatterplot of clusters.
        type (str): type of qt implementation.
    '''
    plot.figure()
    plot.scatter(np.arange(len(labels)), labels, marker = '.',cmap='prism')
    plot.xlabel("Frame Number")
    plot.ylabel("Cluster Index")
    plot.locator_params(axis="both", integer=True, tight=True)
    if preprocess:
        if test:
            plot.title("%s - Scatterplot of clusters vs frame - %s - %s" % (test, preprocess, type))
        else:
            plot.title("Scatterplot of clusters vs frame - %s - %s" % (preprocess, type))
    else:
        if test:
            plot.title("%s - Scatterplot of clusters vs frame - %s" % (test, type))
        else:
            plot.title("Scatterplot of clusters vs frame - %s" % type)
    plot.savefig(DATA + DATA_DEST + dest + "/scatterplot-%s.png" % type, dpi=300)
    if visualise:
        plot.show()


def embedding_plot(cluster_labels, data, dest, type, preprocess, test, visualise):
    fig, ax = plot.subplots()
    clustered = (cluster_labels >= -1)
    embedding = ax.scatter(data[clustered, 0],
                           data[clustered, 1],
                           c=cluster_labels[clustered],
                           s=1,
                           cmap='gist_rainbow',
                           )
    legend = ax.legend(*embedding.legend_elements(), title='Clusters', bbox_to_anchor=(1.2, 1.2))
    ax.add_artist(legend)
    if test:
        plot.title('%s - %s clustering of %s embedding' % (test, type, preprocess))
    else:
        plot.title('%s clustering of %s embedding' % (type, preprocess))
    plot.savefig('%s%s%s/%s-embedding-%s.png' % (DATA, DATA_DEST, dest, preprocess, type), bbox_inches='tight')
    if visualise:
        plot.show()
