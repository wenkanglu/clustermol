import time

import numpy as np
from mdtraj import Trajectory

import umap


def umap_main(input_data, args):
    """
    Function to perform UMAP preprocessing on some data.

    Args:
        input_data (Trajectory/array): Data to be preprocessed by UMAP.
        args (Namespace): User arguments from config file or argparser.

    Returns:
        embedding: UMAP-preprocessed data with reduced dimensions.
    """

    umap_cluster = umap.UMAP(n_components=int(args.ncomponents),
                             n_neighbors=int(args.nneighbours),
                             min_dist=0.0,
                             # seed to get same embedding for same configuration on same data
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

    return embedding
