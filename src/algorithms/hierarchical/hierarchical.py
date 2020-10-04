import os
import numpy as np
import mdtraj
import unittest
import scipy.cluster.hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, ClusterWarning
import matplotlib.pyplot as plot
from sklearn import cluster, datasets
from itertools import cycle, islice
from main.constants import DATA, DATA_DEST, ANGSTROM
from processing import post_proc
from warnings import simplefilter
simplefilter("ignore", ClusterWarning) # Used to ignore warning for test cases

clustering_type = ["single", "complete", "average", "ward"]

def illustrateRMSD(rmsd_matrix, dest):
    '''
    DESCRIPTION
    Creates an RMDS heatmap

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix.
        dest (str): destination to save rmsd matrix visualization.
    '''
    plot.figure()
    plot.title('Pairwise Distance Matrix')
    plot.imshow(rmsd_matrix, cmap='Greys', interpolation='nearest', origin='lower')
    plot.xlabel('Simulation frames')
    plot.ylabel('Simulation frames')
    plot.colorbar()
    plot.savefig(DATA + DATA_DEST + dest + "/RMSD-matrix.png", dpi=300)
    # plot.show()
    plot.close()

def rmsd_stats(rmsd_matrix):
        '''
        DESCRIPTION
        Print statistics from RMSD matrxi.

        Arguments:
            rmsd_matrix (numpy.ndarray): rmsd matrix.
        '''
        print(">>> Max pairwise rmsd (%s): %f" % (ANGSTROM , np.max(rmsd_matrix)))
        print(">>> Average pairwise rmsd (%s): %f" % (ANGSTROM , np.mean(rmsd_matrix)))
        print(">>> Median pairwise rmsd (%s): %f" % (ANGSTROM , np.median(rmsd_matrix)))
        return np.max(rmsd_matrix), np.mean(rmsd_matrix), np.median(rmsd_matrix)

def produceClusters(linkage_matrix, args):
    '''
    DESCRIPTION
    Produces clusters provided linkage matrix and either number of clusters or
    dendrogramdistance

    Arguments:
        linkage_matrix (numpy.ndarray): cluster linkage matrix.
        args (args): arguments of parser for k_clusters or distance.
    Return:
        clusters (numpy.ndarray): list of clusters produced.
    '''
    clusters = -1
    try:
        if args.k_clusters:
            clusters = fcluster(linkage_matrix, int(args.k_clusters), criterion='maxclust')
        elif args.ddistance:
            clusters = fcluster(linkage_matrix, float(args.ddistance), criterion='distance')
        else:
            print("Argument Error (Hierarchical) - Invalid Selection")
        return clusters
    except ValueError:
        print("Error with in producing hierarchical clusters please check arguemnts")
        sys.exit(1)


def produceClustersV(linkage_matrix, k):
    '''
    DESCRIPTION
    Produces clusters provided linkage matrix and either number of clusters
    for test case.

    Arguments:
        linkage_matrix (numpy.ndarray): cluster linkage matrix.
        k (int): number of clusters to produce.
    Return:
        clusters (numpy.ndarray): list of clusters produced.
    '''
    return fcluster(linkage_matrix, k, criterion='maxclust')

def save_dendrogram(linkage_type, linkage_matrix, dest):
    '''
    DESCRIPTION
    Popup and/or Dendrogram produced by hierarchical clustering.

    Arguments:
        linkage_type (str): string for hierarchical type.
        linkage_matrix (numpy.ndarray): linkage matrix from clustering.
        dest (str): destination to save dendrogram visualization.
        flag_display (bool): flat to display dendrogram as popup.
    '''
    plot.title('Dendrogram for %s linkage hierarchical clustering' %linkage_type)
    _ = scipy.cluster.hierarchy.dendrogram(
    linkage_matrix.astype("float64"),
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    show_contracted=True,
    color_threshold=0,
    above_threshold_color='k',
    show_leaf_counts=True # to get a distribution impression in truncated branches
    )
    axes = plot.gca()
    ymin, ymax = axes.get_ylim()
    # plot.axhline(y=ymax*2/3, c='k') # Adjust Cut-off line here
    plot.xticks(fontsize=6)
    plot.xlabel('Frame Count')
    plot.ylabel('Distance')
    # plt.text(0.50, 0.02, "Text relative to the AXES centered at : (0.50, 0.02)", transform=plt.gca().transAxes, fontsize=14, ha='center', color='blue')
    # plot.text(0.8, 0.8, 'Cut-off line [Adjust in Code]', style='italic',ha='left',transform=plot.gca().transAxes,
    #     bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 4})
    plot.savefig(DATA + DATA_DEST + dest + "/dendrogram-clustering-%s.png" % linkage_type, dpi=300)
    plot.close()

def preprocessing_hierarchical(traj):
    '''
    DESCRIPTION
    Preprocessing required for Hierarchical clustering. Calculates RMDS Matrix and
    converts it to squareform. Use ANGSTROMs to keep consistant.

    Arguments:
        traj (mdtraj.Trajectory): trajectory object from MDTraj libary.
    Return:
        rmsd_matrix (numpy.np): rmsd matrix for clustering.
    '''
    rmsd_matrix = np.ndarray((traj.n_frames, traj.n_frames), dtype=np.float64)
    for i in range(traj.n_frames):
        rmsd_ = mdtraj.rmsd(traj, traj, i, parallel=True)*10 # x10 for ANGSTROMs
        rmsd_matrix[i] = rmsd_
    print('>>> RMSD matrix complete')
    # assert np.all(rmsd_matrix - rmsd_matrix.T < 1e-6) # Need to figure out what this is for.
    reduced_distances = squareform(rmsd_matrix, checks=False)
    return reduced_distances

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

def clustering(rmsd_matrix, linkage_type):
    '''
    DESCRIPTION
    Runs Hierarchical clustering methods from scipy libary.

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix used for clustering.
        linkage_type (str): string for hierarchical type.
    Return:
        linkage_matrix (numpy.ndarray): cluster linkage matrix.
    '''
    print(">>> Performing %s based Hierarchical clustering " % linkage_type)
    assert(linkage_type in clustering_type)
    simplefilter("ignore", ClusterWarning)
    linkage_matrix = scipy.cluster.hierarchy.linkage(rmsd_matrix, method=linkage_type)
    print(">>> Hierarchical clustering (%s) complete " %linkage_type)
    return linkage_matrix

def runHierarchicalClustering(traj, args):
    '''
    DESCRIPTION
    Method for running Hierarchical clustering algorithm bases on type. Type
    refers to linkage factor. Filename and destintion are also used for input
    files and destination of desired output.

    Arguments:
        traj (mdtraj.traj): data used for clustering (test cases will not be Trajectory object)
        linkage_type (str): string for hierarchical type.
    '''
    clusters = -1
    if isinstance(traj, mdtraj.Trajectory):
        traj = clean_trajectory(traj)
        rmsd_matrix_temp = preprocessing_hierarchical(traj)
        rmsd_stats(rmsd_matrix_temp)
        linkage_matrix = clustering(rmsd_matrix_temp, args.linkage)
        post_proc.cophenetic(linkage_matrix, rmsd_matrix_temp)
        clusters = produceClusters(linkage_matrix, args)

        if (args.visualise):
            save_dendrogram(args.linkage, linkage_matrix, args.destination) # False not to show dendrogram
            illustrateRMSD(squareform(rmsd_matrix_temp, checks=False), args.destination)
    else:
        data = traj
        pairwise_distance = pdist(data, metric="euclidean")
        reduced_distances = squareform(pairwise_distance, checks=True)
        rmsd_stats(reduced_distances)
        linkage_matrix = clustering(reduced_distances, args.linkage)
        post_proc.cophenetic(linkage_matrix, pairwise_distance)
        clusters = produceClusters(linkage_matrix, args)
        if(args.visualise):
            # illustrateRMSD(reduced_distances, args.destination) # outputs rmsd matrix to destintion
            post_proc.plotTestData(data,clusters, args.destination)
            save_dendrogram(args.linkage, linkage_matrix, args.destination) # False not to show dendrogram

    return clusters

class QTTest(unittest.TestCase):
    # Test RMSD Stats produce epxcted output.
    def testRmsdTest(self):
        testMatrix = np.empty(10)
        testMatrix.fill(1)
        self.assertEqual(rmsd_stats(testMatrix), (1,1,1))
    # Test our implemnataion of preprocessing for rmsd vs random online dataset/traj.
    def testRmsdTraj(self):
        traj = mdtraj.load('test_data/ala2.h5')
        distances = np.empty((traj.n_frames, traj.n_frames), dtype=np.float64)
        for i in range(traj.n_frames):
            d_ = mdtraj.rmsd(traj, traj, i, parallel=True)*10  # x10 for ANGSTROMs
            distances[i] = d_
        # self.assertAlmostEqual(preprocessing_qt(traj), distances, places=7)
        np.testing.assert_almost_equal(preprocessing_hierarchical(traj), squareform(distances, checks=False), decimal=5)

    # Testing clustering produces results
    # Tests both clustering and produce clusters.
    # Minus one is for indexing, clusters produce from zero while index from blobs is -1
    # Ward Method
    def testClustHW(self):
        blobs = datasets.make_blobs(n_samples=100, centers=5, random_state=3, cluster_std =0.2, center_box=(0, 10))
        input_data, y = blobs
        rmsd_matrix = squareform(pdist(input_data, metric="euclidean"), checks=True)
        kclust= 5
        self.assertCountEqual(produceClustersV(clustering(rmsd_matrix, "ward"), 5)-1,y)
    # Single Method
    def testClustHS(self):
        blobs = datasets.make_blobs(n_samples=100, centers=5, random_state=3, cluster_std =0.2, center_box=(0, 10))
        input_data, y = blobs
        rmsd_matrix = squareform(pdist(input_data, metric="euclidean"), checks=True)
        kclust= 5
        self.assertCountEqual(produceClustersV(clustering(rmsd_matrix, "single"), 5)-1,y)
    # Complete Method
    def testClustHC(self):
        blobs = datasets.make_blobs(n_samples=100, centers=5, random_state=3, cluster_std =0.2, center_box=(0, 10))
        input_data, y = blobs
        rmsd_matrix = squareform(pdist(input_data, metric="euclidean"), checks=True)
        kclust= 5
        self.assertCountEqual(produceClustersV(clustering(rmsd_matrix, "complete"), 5)-1,y)
        # Average Method
    def testClustHA(self):
        blobs = datasets.make_blobs(n_samples=100, centers=5, random_state=3, cluster_std =0.2, center_box=(0, 10))
        input_data, y = blobs
        rmsd_matrix = squareform(pdist(input_data, metric="euclidean"), checks=True)
        kclust= 5
        self.assertCountEqual(produceClustersV(clustering(rmsd_matrix, "average"), 5)-1,y)



def validation():
    # Old not used anymore.
    # Validation methods can be run with clustermol.
    data_set_size = 3
    n_samples = 500    # Sample szie of 10 000.
    random_state = 3
    centres = 16
    center_box = (0, 10)
    cluster_std = [5.0, 2.5, 0.5, 1.0, 1.1, 2.0, 1.0, 1.0, 2.5, 0.5, 0.5, 0.7, 1.2, 0.2, 1.0, 3.0]
    blobs = datasets.make_blobs(n_samples=n_samples, centers =centres, random_state=random_state, cluster_std =0.2, center_box=center_box)
    no_structure = np.random.rand(n_samples, random_state), None
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

    linkage_matrix = clustering(reduced_distances_b, "single")
    clusters_b_single = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_b, "complete")
    clusters_b_complete = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_b, "average")
    clusters_b_average = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_b, "ward")
    clusters_b_ward = produceClustersTest(linkage_matrix, 16)

    linkage_matrix = clustering(reduced_distances_s, "single")
    clusters_s_single = produceClustersTest(linkage_matrix, 10)
    linkage_matrix = clustering(reduced_distances_s, "complete")
    clusters_s_complete = produceClustersTest(linkage_matrix, 10)
    linkage_matrix = clustering(reduced_distances_s, "average")
    clusters_s_average = produceClustersTest(linkage_matrix, 10)
    linkage_matrix = clustering(reduced_distances_s, "ward")
    clusters_s_ward = produceClustersTest(linkage_matrix, 10)

    linkage_matrix = clustering(reduced_distances_v, "single")
    clusters_v_single = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_v, "complete")
    clusters_v_complete = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_v, "average")
    clusters_v_average = produceClustersTest(linkage_matrix, 16)
    linkage_matrix = clustering(reduced_distances_v, "ward")
    clusters_v_ward = produceClustersTest(linkage_matrix, 16)

    # lb_b1 = qt_vector(reduced_distances_b, 500, 0.6, 10).astype('int')
    # lb_s1 = qt_vector(reduced_distances_s, 500, 1.3, 20)
    # lb_v1 = qt_vector(reduced_distances_v, 500, 2.2, 20)
    #
    # lb_b2 = qt_orginal(reduced_distances_b, 500, 0.6, 10).astype('int')
    # lb_s2 = qt_orginal(reduced_distances_s, 500, 1.3, 20)
    # lb_v2 = qt_orginal(reduced_distances_v, 500, 2.2, 20)
    # colors = numpy.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                          '#f781bf', '#a65628', '#984ea3',
    #                                          '#999999', '#e41a1c', '#dede00']),
    #                                   int(max(lb_b2) + 1))))
    # # add black color for outliers (if any)
    # colors = numpy.append(colors, ["#000000"])
    plot.figure(figsize=(8, 6))
    plot.subplots_adjust(left=.1, right=.98, bottom=.05, top=.96, wspace=.2,
                    hspace=.2)
    plot.subplot(data_set_size, len(clustering_type), 1)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], c =clusters_b_single, cmap =plot.cm.tab20 )
    plot.subplot(data_set_size, len(clustering_type), 2)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], c =clusters_b_complete, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 3)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], c =clusters_b_average, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 4)
    plot.scatter(Xblobs[:, 0], Xblobs[:, 1], c =clusters_b_ward, cmap =plot.cm.tab20)

    plot.subplot(data_set_size, len(clustering_type), 5)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c =clusters_s_single, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 6)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c =clusters_s_complete, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 7)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c =clusters_s_average, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 8)
    plot.scatter(Xno_structure[:, 0], Xno_structure[:, 1], c =clusters_s_ward, cmap =plot.cm.tab20)

    plot.subplot(data_set_size, len(clustering_type), 9)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =clusters_v_single, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 10)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =clusters_v_complete, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 11)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =clusters_v_average, cmap =plot.cm.tab20)
    plot.subplot(data_set_size, len(clustering_type), 12)
    plot.scatter(Xvaried_blobs[:, 0], Xvaried_blobs[:, 1], c =clusters_v_ward, cmap =plot.cm.tab20)


    plot.show()
    # print(numpy.ndim(reduced_distances))
    # illustrateRMSD(reduced_distances_b, "test_validation")
    # lb = qt_vector(reduced_distances, 150, 1.3, 20)

if __name__ == "__main__":
    # unittest.main()
    print("run clustermol -h for information how to use this framework.")
