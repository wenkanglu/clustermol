import time

from sklearn.manifold import TSNE
from mdtraj import Trajectory
import numpy as np


def tsne_main(input_data, args):
    """
    Function to perform t-SNE preprocessing on some data.

    Args:
        input_data (Trajectory/array): Data to be preprocessed by t-SNE.
        args (Namespace): User arguments from config file or argparser.

    Returns:
        embedding: t-SNE-preprocessed data with reduced dimensions.
    """

    embedding = None
    start_time = time.time()
    tsne = TSNE(n_components=int(args.ncomponents), perplexity=int(args.nneighbours), n_iter=5000, learning_rate=200)

    if isinstance(input_data, Trajectory):
        coords = np.reshape(input_data.xyz, (input_data.n_frames, 3 * input_data.n_atoms))
        print("trajectory loaded")
        embedding = tsne.fit_transform(coords)

    else:
        print("data loaded")
        embedding = tsne.fit_transform(input_data)

    tsne_time = time.time()
    print("--- %s seconds to perform t-SNE---" % (tsne_time - start_time))

    return embedding
