import mdtraj as md
import numpy as np
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

filename = "MenW_6RU_0_to_100ns.pdb"
save_location = "/"
clustering_type = ["single", "complete", "average", "ward"]

# Load Trajectory
trajectory = md.load(filename)

# Illustrate that all Frames are loaded
print(">>> File Loaded <<<")
print(trajectory)

# Calculate RMSD Pairwsie Matrix
rmsd_matrix = np.ndarray((trajectory.n_frames, trajectory.n_frames), dtype=np.float16)
for i in range(trajectory.n_frames):
    rmsd_ = md.rmsd(trajectory, trajectory, i) #currently we assume they are pre-centered, bat can they not be?
    rmsd_matrix[i] = rmsd_
print('Max pairwise rmsd: %f nm' % np.max(rmsd_matrix))
print('>>> RMSD matrix completed <<<')

# Clean up and Preprocessing of Matrix
#assert np.all(rmsd_matrix - rmsd_matrix.T < 1e-6)
reduced_distances = squareform(rmsd_matrix, checks=False)

def export_dendrogram(hierarchical_type, linkage):
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True)
    plot.savefig("graphics/clustering-%s.png" % hierarchical_type)

# Method used to run specific type of hierarchical clustering, based on users choices.
def cluserting(hierarchical_type):
    if hierarchical_type == clustering_type[0]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[0])
        linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method=clustering_type[0])
        export_dendrogram(hierarchical_type, linkage)
    elif hierarchical_type == clustering_type[1]:
        print('>>> Performing %s basd Hierarchical clustering <<<' % clustering_type[1])
        linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method=clustering_type[1])
        export_dendrogram(hierarchical_type, linkage)
    elif hierarchical_type == clustering_type[2]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[2])
        linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method=clustering_type[2])
        export_dendrogram(hierarchical_type, linkage)
    elif hierarchical_type == clustering_type[3]:
        print('>>> Performing %s based Hierarchical clustering <<<' % clustering_type[3])
        linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method=clustering_type[3])
        export_dendrogram(hierarchical_type, linkage)

cluserting("ward")
