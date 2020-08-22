import os
import numpy
import preprocessing
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

clustering_type = ["single", "complete", "average", "ward"]

def show_dendrogram(hierarchical_type, linkage):
    '''
    DESCRIPTION
    Popup Dendrogram produced by hierarchical clustering.

    Arguments:
        hierarchical_type (str): string for hierarchical type .
        linkage (numpy.ndarray): linkage matrix from clustering.
    '''
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage.astype("float64"), no_labels=True)
    plot.show()

def save_dendrogram(hierarchical_type, linkage, destination):
    '''
    DESCRIPTION
    Save Dendrogram produced by hierarchical clustering.

    Arguments:
        rmsd_matrix_temp (numpy.ndarray): rmsd matrix used for clustering.
        hierarchical_type (str): string for hierarchical type.
        destination (str): string for file location within data_dest folder
    '''
    os.chdir(os.path.join(os.path.dirname(__file__))+ "data/data_dest/")  # changes cwd to always be at clustermol
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True)
    plot.savefig("dendrogram-clustering-%s.png" % hierarchical_type)


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
    elif hierarchical_type == clustering_type[1]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[1])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[1])
    elif hierarchical_type == clustering_type[2]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[2])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[2])
    elif hierarchical_type == clustering_type[3]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[3])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[3])
    print(">>> Hierarchical clustering (%s) complete " %hierarchical_type )
    return linkage

def runHierarchicalClustering(filename, destination, type):
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_hierarchical(traj)
    linkage_temp = cluserting(rmsd_matrix_temp, type).astype("float64")
    show_dendrogram(type, linkage_temp)
    save_dendrogram(type, linkage_temp, destination)

if __name__ == "__main__":
    runHierarchicalClustering("MenY_reduced_100_frames.pdb", "graphics", "single")
