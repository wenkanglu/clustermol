import os
import time

import matplotlib
import mdtraj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from warnings import filterwarnings


def cluster(args):
    filterwarnings('ignore')

    start_time = time.time()
    traj = mdtraj.load(os.path.join("data", "data_src", "MenW_ds_10_pf.pdb"))
    traj.remove_solvent(inplace=True)
    print(traj.n_atoms)

    traj_time = time.time()
    print("--- %s seconds to load trajectory---" % (traj_time - start_time))
    coords = np.reshape(traj.xyz, (traj.n_frames, 3*traj.n_atoms))
    # print(coords[0])

    # iris = datasets.load_iris()

    # n_components=7 because we have 7 features in the dataset
    # pca = PCA(n_components=3*traj.n_atoms)
    # pca.fit(coords)
    # variance = pca.explained_variance_ratio_
    # var = np.cumsum(np.round(variance, 3)*100)
    #
    # plt.figure(figsize=(12, 6))
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('# of Features')
    # plt.title('PCA Analysis')
    # plt.ylim(0, 100.5)
    # plt.plot(var)
    # plt.show()

    pca = PCA(n_components=80)
    pca_scale = pca.fit_transform(coords)
    pca_df_scale = pd.DataFrame(pca_scale)
    pca_time = time.time()
    print("--- %s seconds to perform pca---" % (pca_time - traj_time))

    tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=5000, learning_rate=200)
    tsne_scale_results = tsne.fit_transform(pca_df_scale)
    tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2'])
    tsne_time = time.time()
    print("--- %s seconds to perform t-sne---" % (tsne_time - pca_time))

    # plt.figure(figsize=(10, 10))
    # plt.scatter(tsne_df_scale.iloc[:, 0], tsne_df_scale.iloc[:, 1], alpha=0.25, facecolor='lightslategray')
    # plt.xlabel('tsne1')
    # plt.ylabel('tsne2')
    # plt.show()

    sse = []
    k_list = range(1, 15)
    for k in k_list:
        km = KMeans(n_clusters=k)
        km.fit(tsne_df_scale)
        sse.append([k, km.inertia_])
    tsne_results_scale = pd.DataFrame({'Cluster': range(1, 15), 'SSE': sse})
    plt.figure(figsize=(12, 6))
    plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
    plt.title('Optimal Number of Clusters using Elbow Method (tSNE_Scaled Data)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    kmeans_tsne_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(tsne_df_scale)
    labels_tsne_scale = kmeans_tsne_scale.labels_
    clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters': labels_tsne_scale})], axis=1)
    plt.figure(figsize=(15, 15))
    sns.scatterplot(clusters_tsne_scale.iloc[:, 0], clusters_tsne_scale.iloc[:, 1], hue=labels_tsne_scale, palette='Set1',
                    s=100, alpha=0.6).set_title('Cluster Vis tSNE Scaled Data', fontsize=15)
    plt.legend()
    plt.show()

    # Scene = dict(xaxis=dict(title='tsne1'), yaxis=dict(title='tsne2'), zaxis=dict(title='tsne3'))
    # labels = labels_tsne_scale
    # trace = go.Scatter3d(x=clusters_tsne_scale.iloc[:, 0],
    #                      y=clusters_tsne_scale.iloc[:, 1],
    #                      z=clusters_tsne_scale.iloc[:, 2],
    #                      mode='markers', marker=dict(color=labels,
    #                                                  colorscale='Viridis', size=10, line=dict(color='yellow', width=5)))
    #
    # layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=1000, width=1000)
    # data = [trace]
    # fig = go.Figure(data=data, layout=layout)
    # fig.show()
