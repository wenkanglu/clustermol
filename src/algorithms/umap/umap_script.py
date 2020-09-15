import time

import numpy as np
from mdtraj import Trajectory
import seaborn as sns

import umap


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

    return embedding
