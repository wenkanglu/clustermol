import numpy
import numpy.ma as ma
import preprocessing
import postprocessing
import matplotlib.pyplot as plot


def qt_orginal(rmsd_matrix, cutoff, minimum_membership):
    # ---- Delete unuseful values from matrix (diagonal &  x>threshold) -----------
    n_frames =len(rmsd_matrix)  # Number of frames
    rmsd_matrix[rmsd_matrix > cutoff] = numpy.inf # make
    rmsd_matrix[rmsd_matrix == 0] = numpy.inf
    degrees = (rmsd_matrix < numpy.inf).sum(axis=0)
    numpy.set_printoptions(threshold=numpy.inf)
    # print(degrees)

    # =============================================================================
    # QT algotithm
    # =============================================================================

    cluster_labels = numpy.ndarray(n_frames, dtype=numpy.int64) # Frame size needs to change
    cluster_labels.fill(-1)

    cluster_index = 0
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
    postprocessing.scatterplot_single(cluster_labels, n_frames, "QT_original")
    postprocessing.saveClusters(cluster_labels, "QT_original")
    return cluster_labels

def qt_like(rmsd_matrix, cutoff, minimum_membership):
    n_frames = len(rmsd_matrix)  # Number of frames from Trajectory
    # print(rmsd_matrix)
    rmsd_matrix = rmsd_matrix <= cutoff  # Remove all those less than or equal to the cut-off value
    # print(rmsd_matrix)
    centers = []  # Empty centers, cenrtal frame of cluster.
    cluster_index = 0  # Cluster index, used for cluster indexing to frame.
    cluster_labels = numpy.empty(n_frames) # Frame size needs to change
    cluster_labels.fill(numpy.NAN)

    # Looping while cutoff_mask is not empty.
    while rmsd_matrix.any():
        membership = rmsd_matrix.sum(axis=1)
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
    postprocessing.scatterplot_single(cluster_labels, n_frames, "QT_like")
    postprocessing.saveClusters(cluster_labels, "QT_like")
    return cluster_labels

def runQT(filename, destination, type):
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_qt(traj)  # Need to write general pre-process.
    if type == "qt_original":
        qt_orginal(rmsd_matrix_temp, 0.75, 50)
    elif type == "qt_like":
        qt_like(rmsd_matrix_temp, 0.75, 5)
    else:
        lb1 = qt_like(rmsd_matrix_temp, 0.25, 5)
        lb2 = qt_orginal(rmsd_matrix_temp, 0.25, 5)
        postprocessing.scatterplot_multiple(lb1, lb2, len(rmsd_matrix_temp))

if __name__ == "__main__":


    # runQT("MenY_reduced_100_frames.pdb", "data_dest", "qt_original")
    # runQT("MenY_reduced_100_frames.pdb", "data_dest", "qt_like")
    # runQT("MenW_6RU_0_to_10ns.pdb", "data_dest", "")
    # runQT("MenY_reduced_100_frames.pdb", "data_dest", "")
    # runQT("MenY_aligned_downsamp10.pdb", "data_dest", "qt_orginal")
    runQT("MenW_aligned_downsamp10_reduced(Nic).pdb", "data_dest", "qt_like")
