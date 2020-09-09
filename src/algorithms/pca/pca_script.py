import time

import matplotlib.pyplot as plt
from mdtraj import Trajectory
import numpy as np
from sklearn.decomposition import PCA


def pca_main(input_data, args):
    start_time = time.time()

    if isinstance(input_data, Trajectory):
        pca = PCA(n_components=min(30, 3 * input_data.n_atoms))
        coords = np.reshape(input_data.xyz, (input_data.n_frames, 3 * input_data.n_atoms))
        pca_output = pca.fit_transform(coords)
    else:
        pca = PCA(n_components=min(30, len(input_data[0])))
        pca_output = pca.fit_transform(input_data)

    pca_time = time.time()
    print("--- %s seconds to perform PCA---" % (pca_time-start_time))

    if args.visualise:
        variance = pca.explained_variance_ratio_
        var = np.cumsum(np.round(variance, 3)*100)

        plt.figure(figsize=(12, 6))
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(0, 100.5)
        plt.plot(var)
        plt.show()

    return pca_output
