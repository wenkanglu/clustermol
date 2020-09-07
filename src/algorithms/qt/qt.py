import os
import numpy
import mdtraj
import numpy.ma as ma
import matplotlib.pyplot as plot

from main.constants import DATA_SRC, DATA, DATA_DEST


def illustrateRMSD(rmsd_matrix, dest):
    '''
    DESCRIPTION
    Creates an RMDS heatmap

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix.
        dest (str): destination to save rmsd matrix visualization.
    '''
    plot.figure()
    plot.imshow(rmsd_matrix, cmap='viridis', interpolation='nearest')
    print(">>> Max pairwise rmsd: %f nm" % numpy.max(rmsd_matrix))
    print(">>> Average pairwise rmsd: %f nm" % numpy.mean(rmsd_matrix))
    print(">>> Median pairwise rmsd: %f nm" % numpy.median(rmsd_matrix))
    plot.colorbar()
    plot.savefig(DATA + DATA_DEST + dest + "/RMSD-matrix.png", dpi=300)
    # plot.show()
    plot.close()

def rmsd_vs_frame(no_frames, rmsds, dest):
    '''
    DESCRIPTION
    Produce cluster scatter plot of frames. Skips first frame that computed against.

    Arguments:
        clusters_arr (mdtraj.traj): trajectory.
        rmsds (numpy.ndarray): rmsd matrix for visualization.
        dest (str): destination to save rmsd vs frame visualization.
    '''
    plot.figure()
    plot.plot(numpy.arange(1,no_frames), rmsds[1:, ], 'r', label='all atoms')
    plot.legend()
    plot.title('RMSDs over time agaist first frame')
    plot.xlabel('Simulation frames')
    plot.ylabel('RMSD (nm)')
    # os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    plot.savefig(DATA + DATA_DEST + dest + "/rmsd-vs-frame.png", dpi=300)
    # plot.show()
    plot.close()

def saveClusters(clusters_arr, dest, type):
    '''
    DESCRIPTION
    Save cluster indexes to text file

    Arguments:
        clusters_arr (numpy.ndarray): Cluster indexes per frame.
        dest (str): destination to save cluster labels.
        type (str): type of qt implementation.
    '''
    # os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    numpy.savetxt(DATA + DATA_DEST + dest + "/clusters-%s.txt" %type, clusters_arr, fmt='%i')

def scatterplot_cluster(clusters_arr, dest, type):
        '''
        DESCRIPTION
        Produce cluster scatter plot of frames

        Arguments:
            clusters_arr (numpy.ndarray): cluster indexes per frame.
            no_frames (int): number of frames
            dest (str): destination to save scatterplot of clusters.
            type (str): type of qt implementation.
        '''
        plot.figure()
        plot.scatter(numpy.arange(len(clusters_arr)), clusters_arr, marker = '.',cmap='prism')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Index")
        plot.locator_params(axis="both", integer=True, tight=True)
        plot.title("Scatterplot of clusters vs frame - %s" % type)
        # os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig(DATA + DATA_DEST + dest + "/scatterplot-%s.png" % type, dpi=300)
        plot.show()
        plot.close()

def getRMSDvsFirstFrame(traj):
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

def qt_orginal(rmsd_matrix, no_frames, cutoff, minimum_membership):
    '''
    DESCRIPTION
    Quality Threshold Algorithm implemnted by Roy Gonzalez Aleman. Original
    proposed by Heyer et. al. Produces clusters from an RMSD matrix where a minimu,
    membership is imposed and a cutoff value in the clusters is used as a threshold.

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix of all frames.
        no_frames (int): number of frames.
        cutoff (int): threshold values when producing clusters.
        minimum_membership (int): minimum membership in a cluster.
    Return:
        cluster_labels (numpy.ndarray): cleaned trajectory object.
    '''
    # ---- Delete unuseful values from matrix (diagonal &  x>threshold) -----------
    # Removes values greater than the cut-off value.
    # Removes all 0's in the matrix (mainly diagonals)
    rmsd_matrix[rmsd_matrix > cutoff] = numpy.inf # make
    rmsd_matrix[rmsd_matrix == 0] = numpy.inf
    degrees = (rmsd_matrix < numpy.inf).sum(axis=0)
    # numpy.set_printoptions(threshold=numpy.inf)
    # print(rmsd_matrix)
    # print(degrees)

    # =============================================================================
    # QT algotithm
    # =============================================================================
    # Cluster lables for each frame. Initally set to -1 (not apart of a cluster).
    cluster_labels = numpy.ndarray(no_frames, dtype=numpy.int64)
    cluster_labels.fill(-1)

    cluster_index = 0 # Starting index for clusters.
    while True:
        # This while executes for every cluster in trajectory ---------------------
        len_precluster = 0
        while True:
            # This while executes for every potential cluster analyzed ------------
            biggest_node = degrees.argmax()
            precluster = []
            precluster.append(biggest_node)
            candidates = numpy.where(rmsd_matrix[biggest_node] < numpy.inf)[0]
            next_ = biggest_node
            distances = rmsd_matrix[next_][candidates]
            while True:
                # This while executes for every node of a potential cluster -------
                next_ = candidates[distances.argmin()]
                precluster.append(next_)
                post_distances = rmsd_matrix[next_][candidates]
                mask = post_distances > distances
                distances[mask] = post_distances[mask]
                if (distances == numpy.inf).all():
                    break
            degrees[biggest_node] = 0
            # This section saves the maximum cluster found so far -----------------
            if len(precluster) > len_precluster:
                len_precluster = len(precluster)
                max_precluster = precluster
                max_node = biggest_node
                degrees = ma.masked_less(degrees, len_precluster)
            if not degrees.max():
                break
        # General break if minsize is reached -------------------------------------
        if len(max_precluster) < minimum_membership:
            break

        # ---- Store cluster frames -----------------------------------------------
        cluster_labels[max_precluster] = cluster_index
        cluster_index += 1
        # print('>>> Cluster # {} found with {} frames at center {} <<<'.format(
        #       ncluster, len_precluster, max_node))

        # ---- Update matrix & degrees (discard found clusters) -------------------
        rmsd_matrix[max_precluster, :] = numpy.inf
        rmsd_matrix[:, max_precluster] = numpy.inf

        degrees = (rmsd_matrix < numpy.inf).sum(axis=0)
        if (degrees == 0).all():
            break
    return cluster_labels
    # return cluster_labels

def qt_vector(rmsd_matrix, no_frames, cutoff, minimum_membership):
    '''
    DESCRIPTION
    Quality Threshold Algorithm implemnted by Melvin et al. Original
    proposed by Daura et al. Produces clusters from an RMSD matrix where a minimum,
    membership is imposed and a cutoff value in the clusters is used as a threshold.
    - Not a true Quaility Threshold algotithm, seen as a vectorised version.

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix of all frames.
        no_frames (int): number of frames.
        cutoff (int): threshold values when producing clusters.
        minimum_membership (int): minimum membership in a cluster.
    Return:
        cluster_labels (numpy.ndarray): cleaned trajectory object.
    '''
    # print(rmsd_matrix)
    rmsd_matrix = rmsd_matrix <= cutoff  # Remove all those less than or equal to the cut-off value
    # print(rmsd_matrix)
    centers = []  # Empty centers, cenrtal frame of cluster.
    cluster_index = 0  # Cluster index, used for cluster indexing to frame.
    cluster_labels = numpy.empty(no_frames) # Frame size needs to change
    cluster_labels.fill(numpy.NAN)

    # Looping while cutoff_mask is not empty.
    while rmsd_matrix.any():
        membership = rmsd_matrix.sum(axis=0)
        center = numpy.argmax(membership)
        # print(center)
        members = numpy.where(rmsd_matrix[center, :]==True)
        if max(membership) <= minimum_membership:
            cluster_labels[numpy.where(numpy.isnan(cluster_labels))] = -1
            break
        cluster_labels[members] = cluster_index
        centers.append(center)
        rmsd_matrix[members, :] = False
        rmsd_matrix[:, members] = False
        cluster_index = cluster_index + 1
        # print(membership)
    return cluster_labels

def cluster(traj, type, args):
    '''
    DESCRIPTION
    Overall implementation of two diffrent implementations of the Quaility
    Threshold algotithm.

    Arguments:
        traj (mdtraj.traj): trajectory object.
        type (string): type of Quaility Threshold algotithm to implemnt.
    '''
    traj = clean_trajectory(traj)
    rmsd_matrix_temp = preprocessing_qt(traj)  # Need to write general pre-process.
    no_frames = traj.n_frames
    illustrateRMSD(rmsd_matrix_temp, args.destination)
    rmsd_vs_frame(no_frames, getRMSDvsFirstFrame(traj), args.destination)
    if type == "qt_original":
        cluster_labels = qt_orginal(rmsd_matrix_temp, no_frames, float(args.qualitythreshold), int(args.minsamples))
        scatterplot_cluster(cluster_labels, args.destination, args.algorithm)
        saveClusters(cluster_labels, args.destination, args.algorithm)
    elif type == "qt_vector":
        cluster_labels = qt_vector(rmsd_matrix_temp, no_frames, float(args.qualitythreshold), int(args.minsamples))
        scatterplot_cluster(cluster_labels, args.destination, args.algorithm)
        saveClusters(cluster_labels, args.destination, args.algorithm)
    else:
        pass

# def runVMD_RMSD_QT(filename, type):
#     '''
#     DESCRIPTION
#     Overall implementation of two diffrent implementations of the Quaility
#     Threshold algotithim using pre-processed rmsd matrix from VMD.
#
#     Arguments:
#         filename (string): filename of .dat VMD matrix file.
#         type (string): type of Quaility Threshold algotithm to implemnt.
#     '''
#     no_frames = int(input(">>> Please enter the number of frames for the RMSD Matrix .dat file\n"))
#     rmsd_matrix_temp = preprocessing.VMD_RMSD_matrix(filename, no_frames)
#     postprocessing.illustrateRMSD(rmsd_matrix_temp)
#     postprocessing.rmsd_vs_frame(no_frames, rmsd_matrix_temp[0])
#     cutoff, min = getArguments()
#     if type == "qt_original":
#         qt_orginal(rmsd_matrix_temp, no_frames, cutoff, min)
#     elif type == "qt_vector":
#         qt_vector(rmsd_matrix_temp, no_frames, cutoff, min)
#     else:
#         pass

# def validation():
#     # The iris dataset is available from the sci-kit learn package
#     n_samples =2000
#     blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#     X,y = blobs
#     d = pdist(X, metric="euclidean")
#     iris = datasets.load_iris()
#     d = pdist(X=iris.data[:, [0, 3]], metric="euclidean")
#     reduced_distances = squareform(d, checks=True)
#     # print(numpy.ndim(reduced_distances))
#     illustrateRMSD(reduced_distances)
#     # Perform agglomerative hierarchical clustering
#     # Use 'average' link function
#     # linkage_temp = cluserting(reduced_distances, type)
#     no_frames = 2000
#     lb = qt_vector(reduced_distances, 150, 1.3, 20)
#     # plot.figure(figsize=(10, 8))
#     #clusters = fcluster(linkage_temp, 3, criterion='maxclust')
#     plot.scatter(iris.data[:,0],iris.data[:,3], c=lb, cmap='viridis')  # plot points with cluster dependent colors
#     plot.show()
#     # Print the first 6 rows
#     # Sepal Length, Sepal Width, Petal Length, Petal Width
#     # print(iris.data)
#     #

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(os.path.join(os.path.dirname(__file__), '..') + DATA + "/" + DATA_SRC)
    traj = mdtraj.load("MenW_0_to_1000ns_aligned(100skip).pdb")
    cluster(traj, "qt_original")
    # runVMD_RMSD_QT("trajrmsd_menW_nic_test.dat", "qt_original")
