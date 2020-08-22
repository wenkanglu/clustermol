import os
import numpy
import preprocessing
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

clustering_type = ["single", "complete", "average", "ward"]

def show_dendrogram(hierarchical_type, linkage):
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage.astype("float64"), no_labels=True)
    plot.show()

def saveData(hierarchical_type, linkage, destination):
    print(os.getcwd())
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # changes cwd to always be at clustermol
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage.astype("float64"), no_labels=True)
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
        print(">>> Performing %s basd Hierarchical clustering " % clustering_type[1])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[1])
    elif hierarchical_type == clustering_type[2]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[2])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[2])
    elif hierarchical_type == clustering_type[3]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[3])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[3])
    print(linkage)
    return linkage

def runHierarchicalClustering(filename, destination, type, visualise):
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_hierarchical(traj)
    linkage_temp = cluserting(rmsd_matrix_temp, type).astype("float64")
    print(linkage_temp)
    show_dendrogram(linkage_temp, type)

if __name__ == "__main__":
    runHierarchicalClustering("MenY_reduced_100_frames.pdb", "graphics", "single", True)
