import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from main.constants import SILHOUETTE, DAVIESBOULDIN, CALINSKIHARABASZ, DATA_DEST, DATA

def label_counts(labels, dest=None):
    unique, counts = np.unique(labels, return_counts=True)
    d = dict(zip(unique, counts))
    if dest:
        with open (DATA + DATA_DEST + dest + '_results.txt', 'a') as f:
            f.write("total frames: {0}\n".format(len(labels)))
            f.write("cluster label: frame count\n")
            for k in d.keys():
                f.write("{0}: {1}\n".format(k, d[k]))
    return d

def calculate_CVI(indices, data, labels, dest):
    with open (DATA + DATA_DEST + dest + '_results.txt', 'a') as f:
        if SILHOUETTE in indices:
            sample_size = 10000 if data.shape[0] > 10000 else None
            f.write("Silhouette score is {0}\n".format(silhouette_score(data, labels, sample_size=sample_size)))
        if DAVIESBOULDIN in indices:
            f.write("Davies-Bouldin score is {0}\n".format(davies_bouldin_score(data, labels)))
        if CALINSKIHARABASZ in indices:
            f.write("Calinski and Harabasz score is {0}\n".format(calinski_harabasz_score(data, labels)))

def save_largest_clusters(n, traj, labels, dest): #note not thoroughly tested
    #check is a trajectory! TODO
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
        trajectories[i].save_pdb(DATA + DATA_DEST + dest + "_cluster" + str(k) + ".pdb")

def save_without_noise(traj, labels, dest): #note not tested
    #check is a trajectory! TODO
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
    noiseless.save_pdb(DATA + DATA_DEST + dest + "_nonoise" + str(k) + ".pdb")# TODO smarter naming
