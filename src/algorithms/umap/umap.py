import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdtraj
import seaborn as sns

import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def umap_main(args):
    traj = mdtraj.load(os.path.join("data", "data_src", "MenW_ds_10_pf.pdb"))
    # traj.remove_solvent(inplace=True)
    traj = traj.atom_slice(traj.topology.select("resname != SOD and type != H"))
    print(traj.n_atoms)
    coords = np.reshape(traj.xyz, (traj.n_frames, 3 * traj.n_atoms))
    print("traj loaded")

    sns.set(style='white', rc={'figure.figsize': (10, 8)})
    umap_cluster = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
    standard_embedding = umap_cluster.fit_transform(coords)
    # print(len(standard_embedding))
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=2)
    plt.show()

    labels = hdbscan.HDBSCAN(
        min_samples=1,
        min_cluster_size=120,
    ).fit_predict(standard_embedding)

    clustered = (labels >= 0)
    plt.scatter(standard_embedding[~clustered, 0],
                standard_embedding[~clustered, 1],
                c=(0.5, 0.5, 0.5),
                s=1,
                alpha=0.5)
    plt.scatter(standard_embedding[clustered, 0],
                standard_embedding[clustered, 1],
                c=labels[clustered],
                s=1,
                cmap='Spectral')
    plt.show()

