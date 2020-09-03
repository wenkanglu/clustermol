import numpy
import numpy.ma as ma
import preprocessing
import postprocessing
import matplotlib.pyplot as plot
from sklearn import cluster, datasets, mixture
from scipy.spatial.distance import pdist, squareform
import sys


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
    postprocessing.scatterplot_cluster(cluster_labels, no_frames, "qt_original")
    postprocessing.saveClusters(cluster_labels, "qt_original")
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
    postprocessing.scatterplot_cluster(cluster_labels, no_frames, "qt_vector")
    postprocessing.saveClusters(cluster_labels, "qt_vector")
    # return cluster_labels

def getArguments():
    '''
    DESCRIPTION
    Gets the cutoff/threshold value needed for Quality Threshold Algorithm.
    In addtion requires a minimum_membership value.

    Returns:
        cutoff (float): threshold value for QT algotithm.
        minimum_membership (int): int value for minimum cluster size.
    '''
    user_input = input(">>> Please enter a cutoff value and minimum membership value\n") or "0.5 10"
    cutoff, min = user_input.split()
    return float(cutoff), int(min)

def cluster(traj, type):
    '''
    DESCRIPTION
    Overall implementation of two diffrent implementations of the Quaility
    Threshold algotithm.

    Arguments:
        traj (mdtraj.traj): trajectory object.
        type (string): type of Quaility Threshold algotithm to implemnt.
    '''
    # traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_qt(traj)  # Need to write general pre-process.
    no_frames = preprocessing.getNumberOfFrames(traj)
    postprocessing.illustrateRMSD(rmsd_matrix_temp)
    postprocessing.rmsd_vs_frame(no_frames, preprocessing.getRMSDvsFirstFrame(traj))
    cutoff, min = getArguments()
    if type == "qt_original":
        qt_orginal(rmsd_matrix_temp, no_frames, cutoff, min)
    elif type == "qt_vector":
        qt_vector(rmsd_matrix_temp, no_frames, cutoff, min)
    else:
        pass

def runVMD_RMSD_QT(filename, type):
    '''
    DESCRIPTION
    Overall implementation of two diffrent implementations of the Quaility
    Threshold algotithim using pre-processed rmsd matrix from VMD.

    Arguments:
        filename (string): filename of .dat VMD matrix file.
        type (string): type of Quaility Threshold algotithm to implemnt.
    '''
    no_frames = int(input(">>> Please enter the number of frames for the RMSD Matrix .dat file\n"))
    rmsd_matrix_temp = preprocessing.VMD_RMSD_matrix(filename, no_frames)
    postprocessing.illustrateRMSD(rmsd_matrix_temp)
    postprocessing.rmsd_vs_frame(no_frames, rmsd_matrix_temp[0])
    cutoff, min = getArguments()
    if type == "qt_original":
        qt_orginal(rmsd_matrix_temp, no_frames, cutoff, min)
    elif type == "qt_vector":
        qt_vector(rmsd_matrix_temp, no_frames, cutoff, min)
    else:
        pass

def validation():
    # The iris dataset is available from the sci-kit learn package
    n_samples =2000
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    X,y = blobs
    d = pdist(X, metric="euclidean")
    iris = datasets.load_iris()
    d = pdist(X=iris.data[:, [0, 3]], metric="euclidean")
    reduced_distances = squareform(d, checks=True)
    # print(numpy.ndim(reduced_distances))
    postprocessing.illustrateRMSD(reduced_distances)
    # Perform agglomerative hierarchical clustering
    # Use 'average' link function
    # linkage_temp = cluserting(reduced_distances, type)
    no_frames = 2000
    lb = qt_vector(reduced_distances, 150, 1.3, 20)
    # plot.figure(figsize=(10, 8))
    #clusters = fcluster(linkage_temp, 3, criterion='maxclust')
    plot.scatter(iris.data[:,0],iris.data[:,3], c=lb, cmap='viridis')  # plot points with cluster dependent colors
    plot.show()
    # Print the first 6 rows
    # Sepal Length, Sepal Width, Petal Length, Petal Width
    # print(iris.data)
    #

if __name__ == "__main__":
    cluster("MenW_0_to_1000ns_aligned(100skip).pdb", "qt_original")
    # runVMD_RMSD_QT("trajrmsd_menW_nic_test.dat", "qt_original")