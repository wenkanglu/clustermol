import os
import numpy
import mdtraj
import numpy.ma as ma
import matplotlib.pyplot as plot
from scipy.spatial.distance import pdist, squareform
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
from main.constants import DATA, DATA_DEST
from mdtraj import Trajectory

directory = os.getcwd()

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
    print(">>> Max pairwise rmsd: %f" % numpy.max(rmsd_matrix))
    print(">>> Average pairwise rmsd: %f" % numpy.mean(rmsd_matrix))
    print(">>> Median pairwise rmsd: %f" % numpy.median(rmsd_matrix))
    plot.xlabel('Simulation frames')
    plot.ylabel('Simulation frames')
    plot.colorbar()
    plot.savefig(directory+DATA + DATA_DEST + dest + "/RMSD-matrix.png", dpi=300)
    # plot.savefig("RMSD-matrix.png", dpi=300)
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
    plot.savefig(directory + DATA + DATA_DEST + dest + "/rmsd-vs-frame.png", dpi=300)
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
    numpy.savetxt(directory + DATA + DATA_DEST + dest + "/clusters-%s.txt" %type, clusters_arr, fmt='%i')

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
        plot.savefig(directory + DATA + DATA_DEST + dest +  "/scatterplot-%s.png" % type, dpi=300)
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
    traj = traj.center_coordinates() # Centre center_coordinates
    traj = traj.remove_solvent()
    return traj

def qt_orginal(rmsd_matrix, no_frames, cutoff, minimum_membership):
    '''
    DESCRIPTION
    Quality Threshold Algorithm implemented by Roy Gonzalez Aleman. Original
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
        print('>>> Cluster # {} found with {} frames'.format(
              cluster_index, len_precluster))

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
    - Runs a vectorized version of QT clustering

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix of all frames.
        no_frames (int): number of frames.
        cutoff (int): threshold values when producing clusters.
        minimum_membership (int): minimum membership in a cluster.
    Return:
        cluster_labels (numpy.ndarray): cleaned trajectory object.
    '''
    rmsd_matrix = rmsd_matrix <= cutoff
    centers = []
    cluster_index = 0
    cluster_labels = numpy.empty(no_frames)
    cluster_labels.fill(numpy.NAN)

    while rmsd_matrix.any():
        membership = rmsd_matrix.sum(axis=1)
        center = numpy.argmax(membership)
        members = numpy.where(rmsd_matrix[center,:]==True)
        if max(membership) <= minimum_membership:
            cluster_labels[numpy.where(numpy.isnan(cluster_labels))] = -1
            break
        cluster_labels[members] = cluster_index
        centers.append(center)
        rmsd_matrix[members,:] = False
        rmsd_matrix[:,members] = False
        cluster_index = cluster_index + 1
        print('>>> Cluster # {} found with {} frames'.format(
              cluster_index, max(membership)))

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
    if isinstance(traj, Trajectory):
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
            print("Invalid Quailty Algorithm selection")
    else:
        print("ToDo test data")
        data = traj

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

def validation():
    algoithm_type = ["qt", "qtvector"]
    data_set_size = 3
    n_samples = 500    # Sample szie of 10 000.
    random_state = 3
    centres = 16
    center_box = (0, 10)
    cluster_std = [5.0, 2.5, 0.5, 1.0, 1.1, 2.0, 1.0, 1.0, 2.5, 0.5, 0.5, 0.7, 1.2, 0.2, 1.0, 3.0]
    blobs = datasets.make_blobs(n_samples=n_samples, centers =centres, random_state=random_state, cluster_std =0.2, center_box=center_box)
    no_structure = numpy.random.rand(n_samples, random_state), None
    varied_blobs = datasets.make_blobs(n_samples=n_samples,
                             centers =centres,
                             cluster_std=cluster_std,
                             random_state=random_state, center_box=center_box)
    Xblobs,yblobs = blobs
    Xno_structure,yno_structure = no_structure
    Xvaried_blobs,yvaried_blobs = varied_blobs

    blobs_distance = pdist(Xblobs, metric="euclidean")
    reduced_distances_b = squareform(blobs_distance, checks=True)
    struct_distance = pdist(Xno_structure, metric="euclidean")
    reduced_distances_s = squareform(struct_distance, checks=True)
    varied_distance = pdist(Xvaried_blobs, metric="euclidean")
    reduced_distances_v = squareform(varied_distance, checks=True)

    lb_b1 = qt_vector(reduced_distances_b, 500, 0.6, 10).astype('int')
    lb_s1 = qt_vector(reduced_distances_s, 500, 1.3, 20)
    lb_v1 = qt_vector(reduced_distances_v, 500, 2.2, 20)

    lb_b2 = qt_orginal(reduced_distances_b, 500, 0.6, 10).astype('int')
    lb_s2 = qt_orginal(reduced_distances_s, 500, 1.3, 20)
    lb_v2 = qt_orginal(reduced_distances_v, 500, 2.2, 20)
    colors = numpy.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(lb_b2) + 1))))
    # add black color for outliers (if any)
    colors = numpy.append(colors, ["#000000"])
    plot.figure(figsize=(8, 6))
    plot.subplots_adjust(left=.1, right=.98, bottom=.05, top=.96, wspace=.2,
                    hspace=.2)
    plot.subplot(data_set_size, len(algoithm_type), 1)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], c =lb_b1, cmap =plot.cm.Set1)
    plot.subplot(data_set_size, len(algoithm_type), 3)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c =lb_s1)
    plot.subplot(data_set_size, len(algoithm_type), 5)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =lb_v1)

    plot.subplot(data_set_size, len(algoithm_type), 2)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], color =colors[lb_b2])
    plot.subplot(data_set_size, len(algoithm_type), 4)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c=lb_s2)
    plot.subplot(data_set_size, len(algoithm_type), 6)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =lb_v2)

    plot.show()
    # print(numpy.ndim(reduced_distances))
    # illustrateRMSD(reduced_distances_b, "test_validation")
    # lb = qt_vector(reduced_distances, 150, 1.3, 20)

if __name__ == "__main__":
    validation()
    # runVMD_RMSD_QT("trajrmsd_menW_nic_test.dat", "qt_original")
