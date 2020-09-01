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
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+"/data/data_src")
    print(">>> Loading file \"%s\"" %filename)
    trajectory = mdtraj.load(filename)
    print(">>> File loaded")
    trajectory = clean_trajectory(trajectory)
    print(">>>", trajectory)
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
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
    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    for i in range(traj.n_frames):
        rmsd_ = mdtraj.rmsd(traj, traj, i, parallel=True)
        rmsd_matrix[i] = rmsd_
    print('>>> RMSD matrix complete')
    # assert numpy.all(rmsd_matrix - rmsd_matrix.T < 1e-6) # Need to figure out what this is for.
    reduced_distances = squareform(rmsd_matrix, checks=False)
    return reduced_distances

def preprocessing_qt(traj):
    '''
    DESCRIPTION
    Preprocessing required for QT clustering. Calculates RMDS Matrix

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for clustering.
    '''
    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    for i in range(traj.n_frames):
        rmsd_ = mdtraj.rmsd(traj, traj, i, parallel=True)
        rmsd_matrix[i] = rmsd_
    print('>>> RMSD matrix complete')
    return rmsd_matrix

def getRMSD_first_frame(traj):
    '''
    DESCRIPTION
    Illustrate change of RMSD over frames with reference to the inital/first frame.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for visualization.
    '''
    rmsd_matrix = numpy.ndarray((traj.n_frames, traj.n_frames), dtype=numpy.float64)
    rmsd_matrix = mdtraj.rmsd(traj, traj, 0)
    return rmsd_matrix

def getNumberOfFrames(traj):
    '''
    DESCRIPTION
    Returns Number of frames within the Trajectory.

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

def VMD_RMSD_matrix(filename, no_frames):
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+"/data/data_src")
    rmsd_matrix = numpy.ndarray((no_frames, no_frames), dtype=numpy.float64)
    rmsd_matrix.fill(0)
    file = open(filename, 'r')
    line = file.readline()
    # print(line)
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file.readline()
        if not line:
            break
        # print(count)
        # print(line)
        if line == '\n':
            break
        a, b, c, d, e = line.split()
        rmsd_matrix[int(b), int(d)] = float(e)
    file.close()
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # change back to clustermol root directory.
    return rmsd_matrix

if __name__ == "__main__":
    print(">>> Preprocessing - Test run")
    # traj = preprocessing_file("MenW_aligned_downsamp10_reduced(Nic).pdb") # Load the file in and format using MDTraj.
    #preprocessing_hierarchical(traj) # Prepares file for MDTraj.
    r = VMD_RMSD_matrix("trajrmsd_menW_nic_test.dat", 401)
    postprocessing.illustrateRMSD(r)
