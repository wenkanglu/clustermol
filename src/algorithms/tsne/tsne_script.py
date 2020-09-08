import os
import time

from mdtraj import Trajectory
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.metrics import silhouette_score
from warnings import filterwarnings


def tsne_main(input_data, args):
    start_time = time.time()
    tsne = TSNE(n_components=int(args.ncomponents), perplexity=int(args.nneighbours), n_iter=5000, learning_rate=200)
    tsne_output = tsne.fit_transform(input_data)
    tsne_time = time.time()
    print("--- %s seconds to perform t-SNE---" % (tsne_time - start_time))
    # tsne_df_scale = pd.DataFrame(tsne_output, columns=['tsne1', 'tsne2'])

    if args.visualise:
        plt.scatter(tsne_output[:, 0], tsne_output[:, 1], s=2)
        plt.show()

        # plt.figure(figsize=(10, 10))
        # plt.scatter(tsne_df_scale.iloc[:, 0], tsne_df_scale.iloc[:, 1], alpha=0.25, facecolor='lightslategray')
        # plt.xlabel('tsne1')
        # plt.ylabel('tsne2')
        # plt.show()

    return tsne_output
