# Nic's Demo
from __future__ import print_function
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
print("START")


traj = md.load("MenY_0_to_1000ns_aligned.pdb")

print("Illustraion with of trajectory loaded - all frames")
print(traj)
print("Illustraion with of trajectory loaded - first 10 frames")
print(traj[0:10])
print("Illustraion with of trajectory loaded - last frame")
print(traj[-1])
print('How many atoms?    %s' % traj.n_atoms)
print('How many residues? %s' % traj.n_residues)
traj[0:100].save_pdb('data_dest/MenY_reduced_100_frames.pdb')

distances = np.empty((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    distances[i] = md.rmsd(traj, traj, i)
print('Max pairwise rmsd: %f nm' % np.max(distances))

#assert np.all(distances - distances.T < 1e-6)
reduced_distances = squareform(distances, checks=False)

linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method="ward")

plt.title('RMSD War linkage hierarchical clustering')
nic = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True)
plt.savefig("graphics/clustering.png")


print("END")
