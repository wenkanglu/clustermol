import mdtraj as md
import numpy as np
import os
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

clustering_type = ["single", "complete", "average", "ward"]

def export_dendrogram(hierarchical_type, linkage, visualise_option, destination):
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+"/data/data_dest/"+destination)  # changes cwd to always be at clustermol
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True)
    plot.savefig("dendrogram-clustering-%s.png" % hierarchical_type)
    if(visualise_option):
        plot.show()

# Method used to run specific type of hierarchical clustering, based on users choices.
def cluserting(hierarchical_type, rmsd_matrix_temp, visualise_option, destination):
    if hierarchical_type == clustering_type[0]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[0])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[0])
        export_dendrogram(hierarchical_type, linkage, visualise_option, destination)
    elif hierarchical_type == clustering_type[1]:
        print('>>> Performing %s basd Hierarchical clustering <<<' % clustering_type[1])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[1])
        export_dendrogram(hierarchical_type, linkage, visualise_option, destination)
    elif hierarchical_type == clustering_type[2]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[2])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[2])
        export_dendrogram(hierarchical_type, linkage, visualise_option, destination)
    elif hierarchical_type == clustering_type[3]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[3])
        linkage = scipy.cluster.hierarchy.linkage(rmsd_matrix_temp, method=clustering_type[3])
        export_dendrogram(hierarchical_type, linkage, visualise_option, destination)

def runClustering(filename, destination, type, visualise):
    rmsd_matrix_temp = preprocessing(filename)
    cluserting(type, rmsd_matrix_temp, visualise, destination)

if __name__ == "__main__":
    runClustering("MenY_reduced_100_frames.pdb", "graphics", "average", True)
