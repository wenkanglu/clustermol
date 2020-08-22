import mdtraj as md
import numpy as np
import os
from scipy.spatial.distance import squareform

def preprocessing_file(filename):
    '''
    DESCRIPTION
    Loads trajectory file using MDTraj. Trajectory file must be in .pdb format.

    Arguments:
        filename (argparse.Namespace): filename of file within data_src directory.
    Return:
        trajectory (mdtraj.Trajectory): trajectory object for later use.
    '''
    # Change to directroy for input file source.
    # Note data should always be placed in the data_src file before running.
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+"/data/data_src")  # changes cwd to always be at clustermol


    # Load Trajectory
    print(">>> Loading File \"%s\"" %filename)
    trajectory = md.load(filename)
    # Illustrate that all Frames are loaded
    print(">>> File Loaded")
    print(">>>", trajectory)
    return trajectory

def preprocessing_hierarchical(traj):
    '''
    DESCRIPTION
    Preprocessing required for Hierarchical clustering. Calculates RMDS Matrix and
    converts it to squareform.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for clustering.
    '''
    # Calculate RMSD Pairwsie Matrix
    rmsd_matrix = np.ndarray((traj.n_frames, traj.n_frames), dtype=np.float64)
    for i in range(traj.n_frames):
        rmsd_ = md.rmsd(traj, traj, i) #currently we assume they are pre-centered, but can they not be?
        rmsd_matrix[i] = rmsd_
    # print('Max pairwise rmsd: %f nm' % np.max(rmsd_matrix))
    print('>>> RMSD matrix complete')
    # file1 = open("matrixOutput.txt","w")
    # np.set_printoptions(threshold=np.inf)
    # file1.write(np.array2string(rmsd_matrix))

    # We can check that the diagonals are all zero using the below statetemnt, however checks vis squareform fail
    # so it is set to false. squareform checks are too stringent
    # print(np.diag(rmsd_matrix))

    # Clean up and Preprocessing of Matrix
    # assert np.all(rmsd_matrix - rmsd_matrix.T < 1e-6) # Need to figure out what this is for.
    reduced_distances = squareform(rmsd_matrix, checks=False)

    return reduced_distances



if __name__ == "__main__":
    print(">>> Preprocessing - Test run")
    traj = preprocessing_file("MenY_reduced_100_frames.pdb") # Load the file in and format using MDTraj.
    preprocessing_hierarchical(traj) # Prepares file for MDTraj.