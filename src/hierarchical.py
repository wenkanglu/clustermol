import os
import numpy
import preprocessing
import postprocessing
import scipy.cluster.hierarchy

clustering_type = ["single", "complete", "average", "ward"]

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
    linkage_temp = cluserting(rmsd_matrix_temp, type).astype("float64")
    postprocessing.show_dendrogram(type, linkage_temp)
    postprocessing.save_dendrogram(type, linkage_temp, destination)

if __name__ == "__main__":
    runHierarchicalClustering("MenY_reduced_100_frames.pdb", "graphics", "ward")
