import mdtraj
import numpy
import postprocessing
import os
from scipy.spatial.distance import squareform


def clean_trajectory(traj):
    '''
    DESCRIPTION
    Takes a trajectory object, removes ions. Other changes to the trajectory can
    be done in this method.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object to be cleaned.
    Return:
        trajectory (mdtraj.Trajectory): cleaned trajectory object.
    '''
    # sel = traj.topology.select("resname != SOD")
    # traj = traj.atom_slice(sel)
    return traj.remove_solvent()

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
    print(">>> Loading file \"%s\"" %filename)
    trajectory = mdtraj.load(filename)
    # Illustrate that all Frames are loaded
    print(">>> File loaded")
    # print(trajectory[::4])
    # trajectory = trajectory[::4]
    # trajectory.save("MenY_aligned_downsamp10_reduced(Nic).pdb")
    # trajectory = clean_trajectory(trajectory) # Cleans trajectory method
    print(">>>", trajectory)
    # print(">>> All atoms: %s" % [atom for atom in trajectory.topology.atoms])
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # change back to clustermol root directory.
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
    ## superpose atomset 1 to reference

    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    for i in range(traj.n_frames):
        rmsd_ = mdtraj.rmsd(traj, traj, i, parallel=True) #currently we assume they are pre-centered, but can they not be?
        rmsd_matrix[i] = rmsd_
    # print('Max pairwise rmsd: %f nm' % np.max(rmsd_matrix))
    print('>>> RMSD matrix complete')
    # postprocessing.illustrateRMSD(rmsd_matrix)
    # postprocessing.illustrateRMSD(rmsd_matrix)
    # file1 = open("matrixOutput.txt","w")
    # np.set_printoptions(threshold=np.inf)
    # file1.write(np.array2string(rmsd_matrix))

    # We can check that the diagonals are all zero using the below statetemnt, however checks vis squareform fail
    # so it is set to false. squareform checks are too stringent
    # print(np.diag(rmsd_matrix))

    # Clean up and Preprocessing of Matrix
    # assert numpy.all(rmsd_matrix - rmsd_matrix.T < 1e-6) # Need to figure out what this is for.
    reduced_distances = squareform(rmsd_matrix, checks=False)
    return reduced_distances
    # return rmsd_matrix

def preprocessing_qt(traj):
    '''
    DESCRIPTION
    Preprocessing required for QT clustering. Calculates RMDS Matrix

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for clustering.
    '''
    # Calculate RMSD Pairwsie Matrix
    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    for i in range(traj.n_frames):
        rmsd_ = mdtraj.rmsd(traj, traj, i, parallel=True, precentered=True) # currently we assume they are pre-centered, but can they not be?
        rmsd_matrix[i] = rmsd_
    # print('Max pairwise rmsd: %f nm' % np.max(rmsd_matrix))
    print('>>> RMSD matrix complete')
    postprocessing.illustrateRMSD(rmsd_matrix)
    # postprocessing.illustrateRMSD(rmsd_matrix)
    return rmsd_matrix

def getRMSD_first_frame(traj):
    '''
    DESCRIPTION
    Illustrate change of RMSD over frames.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for visualization.
    '''
    # Calculate RMSD Pairwsie Matrix
    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    rmsd_matrix = mdtraj.rmsd(traj, traj, 0)
    # print('>>> Single RMSD matrix complete')
    # postprocessing.illustrateRMSD(rmsd_matrix)
    # postprocessing.illustrateRMSD(rmsd_matrix)
    return rmsd_matrix

def numberOfFrames(traj):
    '''
    DESCRIPTION
    Returns Number of frames withing the Trajectory.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        no_frames (int): number of frames from simulation.
    '''
    return traj.n_frames

def getTime(traj):
    '''
    DESCRIPTION
    Returns time period of Trajectory.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        time (int): time period of trajectory.
    '''
    return traj.time

if __name__ == "__main__":
    print(">>> Preprocessing - Test run")
    traj = preprocessing_file("MenW_aligned_downsamp10_reduced(Nic).pdb") # Load the file in and format using MDTraj.
    #preprocessing_hierarchical(traj) # Prepares file for MDTraj.
