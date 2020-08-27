import os
import numpy
import preprocessing
import postprocessing
import scipy.cluster.hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
import matplotlib.pyplot as plot
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

clustering_type = ["single", "complete", "average", "ward"]

def tester(linkage, rmsd_matrix_temp):
    c, coph_dists = cophenet(linkage, pdist(rmsd_matrix_temp))
    print(">>> Cophenetic Distance: %s" % c)

def produceClusters(linkage, no_frames, linkage_type):
    user_input = input("Please enter a cutoff distance value (-d) or number of clusters (-c):\n") or "inconsistent, 3.2"
    type, value = user_input.split()
    if type == "-d":
        clusters = fcluster(linkage, value, criterion='distance')
        postprocessing.scatterplot_time(clusters, no_frames, linkage_type)
        postprocessing.saveClusters(clusters, linkage_type)
    elif type == "-c":
        clusters = fcluster(linkage, value, criterion='maxclust')
        postprocessing.scatterplot_time(clusters, no_frames, linkage_type)
        postprocessing.saveClusters(clusters, linkage_type)
    else:
        print("Default do nothing ATM")
    # print(clusters)


def cut_off_selection():
    # TODO:
    pass


# Method used to run specific type of hierarchical clustering, based on users choices.
def cluserting(rmsd_matrix_temp, hierarchical_type):
    '''
    DESCRIPTION
    Runs Hierarchical clustering methods.

    Arguments:
        rmsd_matrix_temp (numpy.ndarray): rmsd matrix used for clustering.
        hierarchical_type (str): string for hierarchical type.
    Return:
        linkage (numpy.ndarray): cluster linkage.
    '''
    if hierarchical_type == clustering_type[0]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[0])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[0])
        # tester(linkage, rmsd_matrix_temp)
    elif hierarchical_type == clustering_type[1]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[1])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[1])
        # tester(linkage, rmsd_matrix_temp)
    elif hierarchical_type == clustering_type[2]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[2])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[2])
        # tester(linkage, rmsd_matrix_temp)
    elif hierarchical_type == clustering_type[3]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[3])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[3])
        # tester(linkage, rmsd_matrix_temp)
    print(">>> Hierarchical clustering (%s) complete " %hierarchical_type )
    return linkage

def runHierarchicalClustering(filename, destination, type):
    '''
    DESCRIPTION
    Method for running Hierarchical clustering algorithm bases on type. Type
    refers to linkage factor. Filename and destintion are also used for input
    files and destination of desired output.

    Arguments:
        filename (str): string of filename.
        destination (str): string for output destination.
        type (str): string for hierarchical type.
    '''
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_hierarchical(traj)
    linkage_temp = cluserting(rmsd_matrix_temp, type)
    no_frames = preprocessing.numberOfFrames(traj)
    postprocessing.show_dendrogram(type, linkage_temp)
    postprocessing.save_dendrogram(type, linkage_temp, destination)
    produceClusters(linkage_temp, no_frames, type)


def randomDataValidation():
    '''
    DESCRIPTION
    Method for validation with random shapes of clusters for baseline test of algorithms.
    '''
    # Random seed
    numpy.random.seed(0)
    n_samples = 2000  # Sample size - 2000
    # Noisy cirle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
    # Noisy moon data
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    # Blob data
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    # Unstructured data
    no_structure = numpy.random.rand(n_samples, 2), None
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = numpy.dot(X, transformation)
    aniso = (X_aniso, y)
    # Blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

    X, y = aniso
    plot.scatter(X[:, 0], X[:, 1])
    plot.show()
    d = pdist(X, metric="euclidean")
    linkage_temp = cluserting(d, "ward")
    postprocessing.show_dendrogram("ward", linkage_temp)

def validation():
    # The iris dataset is available from the sci-kit learn package
    n_samples =2000
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    X,y = blobs
    d = pdist(X, metric="euclidean")

    iris = datasets.load_iris()
    type = "ward"
    destination = "graphics"
    # Plot results
    # print(iris.data[:, 0])

    # plot.scatter(iris.data[:, 3], iris.data[:, 0])
    # plot.xlabel('Petal width (cm)')
    # plot.ylabel('Sepal length (cm)')
    # plot.show()

    # plot.show()
    # Compute distance matrix
    # d = pdist(X=iris.data[:, [0, 3]], metric="euclidean")
    # print(numpy.ndim(d))
    reduced_distances = squareform(d, checks=True)
    # print(numpy.ndim(reduced_distances))
    # postprocessing.illustrateRMSD(reduced_distances)
    # Perform agglomerative hierarchical clustering
    # Use 'average' link function
    linkage_temp = cluserting(reduced_distances, type)
    no_frames = 2000
    postprocessing.show_dendrogram(type, linkage_temp)
    postprocessing.save_dendrogram(type, linkage_temp, destination)
    tester(linkage_temp, reduced_distances)
    produceClusters(linkage_temp, no_frames, type)
    # plot.figure(figsize=(10, 8))
    # # clusters = fcluster(linkage_temp, 5, criterion='maxclust')
    # plot.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    # plot.show()
    # Print the first 6 rows
    # Sepal Length, Sepal Width, Petal Length, Petal Width
    # print(iris.data)
    #

if __name__ == "__main__":
    # runHierarchicalClustering("MenY_aligned_downsamp10_reduced(Nic).pdb", "graphics", "ward")
    # validation()
    # randomDataValidation()
    # runHierarchicalClustering("MenY_aligned_downsamp10_reduced(Nic).pdb", "graphics", "ward")
    validation()
