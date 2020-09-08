import time

import numpy as np
import matplotlib.pyplot as plt
from mdtraj import Trajectory
import seaborn as sns

import umap
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def umap_main(input_data, args):
    # sns.set(style='white', rc={'figure.figsize': (10, 8)})
    umap_cluster = umap.UMAP(n_components=int(args.ncomponents),
                             n_neighbors=int(args.nneighbours),
                             min_dist=0.0,
                             random_state=42,)
    start_time = time.time()
    if isinstance(input_data, Trajectory):
        coords = np.reshape(input_data.xyz, (input_data.n_frames, 3 * input_data.n_atoms))
        print("trajectory loaded")
        embedding = umap_cluster.fit_transform(coords)

    else:
        print("data loaded")
        embedding = umap_cluster.fit_transform(input_data)

    umap_time = time.time()
    print("--- %s seconds to perform UMAP---" % (umap_time - start_time))

    if args.visualise:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2)
        plt.show()

    return embedding

    # labels = hdbscan.HDBSCAN(
    #     min_samples=1,
    #     min_cluster_size=10,
    # ).fit_predict(embedding)
    #
    # print("50: " + str(silhouette_score(embedding, labels=labels)) + " " +
    #       str(calinski_harabasz_score(embedding, labels=labels)) + " " +
    #       str(davies_bouldin_score(embedding, labels=labels)))
    #
    # clustered = (labels >= 0)
    # plt.scatter(embedding[~clustered, 0],
    #             embedding[~clustered, 1],
    #             c=(0.5, 0.5, 0.5),
    #             s=1,
    #             alpha=0.5)
    # plt.scatter(embedding[clustered, 0],
    #             embedding[clustered, 1],
    #             c=labels[clustered],
    #             s=1,
    #             cmap='Set1')
    # plt.show()
