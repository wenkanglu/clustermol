import os
import numpy
import numpy.ma as ma
import preprocessing
import matplotlib.pyplot as plot


def scatterplot(clusters_arr, no_frames, qt_type):
        #Figures
        plot.figure()
        plot.scatter(numpy.arange(no_frames), clusters_arr, marker = '+')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Number")
        plot.title("Scatter Plot - %s" %qt_type )
        #print(os.getcwd())
        os.chdir(os.path.join(os.path.dirname(__file__))+ "data/data_dest/")
        #print(os.getcwd())
        plot.savefig("Scatterplot_%s.png" %qt_type)
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        plot.close()

def saveClusters(clusters_arr, qt_type):
    os.chdir(os.path.join(os.path.dirname(__file__))+ "data/data_dest/")
    numpy.savetxt("Clusters_%s.txt" %qt_type, clusters_arr, fmt='%i')
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))


def qt_orginal(rmsd_matrix, cutoff, minimum_membership):
    # ---- Delete unuseful values from matrix (diagonal &  x>threshold) -----------
    n_frames =len(rmsd_matrix)
    rmsd_matrix[rmsd_matrix > cutoff] = numpy.inf
    rmsd_matrix[rmsd_matrix == 0] = numpy.inf
    degrees = (rmsd_matrix < numpy.inf).sum(axis=0)

    # =============================================================================
    # QT algotithm
    # =============================================================================

    clusters_arr = numpy.ndarray(n_frames, dtype=numpy.int64) # Frame size needs to change
    clusters_arr.fill(-1)

    ncluster = 0
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
        clusters_arr[max_precluster] = ncluster
        ncluster += 1
        print('>>> Cluster # {} found with {} frames at center {} <<<'.format(
              ncluster, len_precluster, max_node))

        # ---- Update matrix & degrees (discard found clusters) -------------------
        rmsd_matrix[max_precluster, :] = numpy.inf
        rmsd_matrix[:, max_precluster] = numpy.inf

        degrees = (rmsd_matrix < numpy.inf).sum(axis=0)
        if (degrees == 0).all():
            break
    scatterplot(clusters_arr, n_frames, "QT_original")
    saveClusters(clusters_arr, "QT_original")

def qt_like(rmsd_matrix, cutoff, minimum_membership):
    n_frames = len(rmsd_matrix)
    cutoff_mask = rmsd_matrix <= cutoff
    rmsd_matrix = None
    centers = []
    cluster = 0
    labels = numpy.empty(n_frames)
    labels.fill(numpy.NAN)

    while cutoff_mask.any():
        membership = cutoff_mask.sum(axis=1)
        center = numpy.argmax(membership)
        members = numpy.where(cutoff_mask[center,:]==True)
        if max(membership) <= minimum_membership:
            labels[numpy.where(numpy.isnan(labels))] = -1
            break
        labels[members] = cluster
        centers.append(center)
        cutoff_mask[members,:] = False
        cutoff_mask[:,members] = False
        cluster = cluster + 1
    scatterplot(labels, n_frames, "QT_like")
    saveClusters(labels, "QT_like")

def runQT(filename, destination, type):
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_qt(traj) # Need to write general pre-process.
    if type == "qt_original":
        qt_orginal(rmsd_matrix_temp, 0.25, 5)
    elif type == "qt_like":
        qt_like(rmsd_matrix_temp, 0.25, 5)
    else:
        pass

if __name__ == "__main__":
    # runQT("MenY_reduced_100_frames.pdb", "data_dest", "qt_original")
    runQT("MenY_reduced_100_frames.pdb", "data_dest", "qt_like")
