import os
import numpy
import mdtraj
import scipy.cluster.hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
import matplotlib.pyplot as plot
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

clustering_type = ["single", "complete", "average", "ward"]

def saveClusters(clusters_arr, type):
    '''
    DESCRIPTION
    Save cluster indexes to text file

    Arguments:
        clusters_arr (numpy.ndarray): Cluster indexes per frame.
        qt_type (str): qt types of implementation.
    '''
    # os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    numpy.savetxt("clusters-%s.txt" %type, clusters_arr, fmt='%i')

def scatterplot_cluster(clusters_arr, no_frames, cluster_type):
        '''
        DESCRIPTION
        Produce cluster scatter plot of frames

        Arguments:
            clusters_arr (numpy.ndarray): cluster indexes per frame.
            no_frames (int): number of frames
            qt_type (str): qt types of implementation.
        '''
        plot.figure()
        plot.scatter(numpy.arange(no_frames), clusters_arr, marker = '.',cmap='prism')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Index")
        plot.locator_params(axis="both", integer=True, tight=True)
        plot.title("Scatterplot of clusters vs frame - %s" %cluster_type )
        # os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig("scatterplot-%s.png" %cluster_type)
        plot.show()
        plot.close()

def cophenetic(linkage_matrix, rmsd_matrix):
    '''
    DESCRIPTION
    Computes Cophenetic distance given linkage and rmsd_matrix.

    Arguments:
        linkage_matrix (numpy.ndarray): cluster linkage.
        rmsd_matrix (numpy.ndarray): pairwise distance matrix.

    Return:
        c (int): cophenetic distance.
    '''
    reduced_distances = squareform(rmsd_matrix, checks=True)
    c, coph_dists = cophenet(linkage_matrix, pdist(reduced_distances))
    print(">>> Cophenetic Distance: %s" % c)

def produceClusters(linkage_matrix, no_frames, args):
    '''
    DESCRIPTION
    Produces scatterplot of clusters as well as saving cluster indexes/labels
    further analysis.

    Arguments:
        linkage_matrix (numpy.ndarray): cluster linkage matrix.
        no_frames (int): number of frames of trajectory.
        linkage_type (string): linkage type.
    '''
    # user_input = input("Please enter a cutoff distance value (-d) or number of clusters (-c):\n") or "inconsistent, 3.2"
    # type, value = user_input.split()
    # if args.ddistance:
    #     print("1")
    #     clusters = fcluster(linkage_matrix, float(args.ddistance), criterion='distance')
    #     scatterplot_cluster(clusters, no_frames, args.linkage)
    #     saveClusters(clusters, args.linkage)
    print(args.linkage)
    print(args.k_clusters)
    if args.k_clusters:
        clusters = fcluster(linkage_matrix, int(args.k_clusters), criterion='maxclust')
        scatterplot_cluster(clusters, no_frames, args.linkage)
        saveClusters(clusters, args.linkage)
    else:
        print("Invalid Selection")

def save_dendrogram(hierarchical_type, linkage, flag_display):
    '''
    DESCRIPTION
    Popup Dendrogram produced by hierarchical clustering.

    Arguments:
        hierarchical_type (str): string for hierarchical type.
        linkage (numpy.ndarray): linkage matrix from clustering.
    '''
    plot.title('Dendrogram for %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(
    linkage.astype("float64"),
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    show_contracted=True,
    show_leaf_counts=True # to get a distribution impression in truncated branches
    )
    axes = plot.gca()
    ymin, ymax = axes.get_ylim()
    plot.axhline(y=ymax*2/3, c='k')
    plot.xlabel('Frame Count')
    plot.ylabel('Distance')
    # plt.text(0.50, 0.02, "Text relative to the AXES centered at : (0.50, 0.02)", transform=plt.gca().transAxes, fontsize=14, ha='center', color='blue')
    plot.text(0.8, 0.8, 'ToDO', style='italic',ha='left',transform=plot.gca().transAxes,
        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 4})
    plot.savefig("dendrogram-clustering-%s.png" % hierarchical_type)
    if flag_display:
        plot.show()
    plot.close()

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

def cluserting(rmsd_matrix, linkage_type):
    '''
    DESCRIPTION
    Runs Hierarchical clustering methods.

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix used for clustering.
        linkage_type (str): string for hierarchical type.
    Return:
        linkage_matrix (numpy.ndarray): cluster linkage matrix.
    '''
    if linkage_type == clustering_type[0]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[0])
        linkage_matrix = scipy.cluster.hierarchy.linkage(rmsd_matrix, method=clustering_type[0])
        cophenetic(linkage_matrix, rmsd_matrix)
    elif linkage_type == clustering_type[1]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[1])
        linkage_matrix = scipy.cluster.hierarchy.linkage(rmsd_matrix, method=clustering_type[1])
        cophenetic(linkage_matrix, rmsd_matrix)
    elif linkage_type == clustering_type[2]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[2])
        linkage_matrix = scipy.cluster.hierarchy.linkage(rmsd_matrix, method=clustering_type[2])
        cophenetic(linkage_matrix, rmsd_matrix)
    elif linkage_type == clustering_type[3]:
        print(">>> Performing %s based Hierarchical clustering " % clustering_type[3])
        linkage_matrix = scipy.cluster.hierarchy.linkage(rmsd_matrix, method=clustering_type[3])
        cophenetic(linkage_matrix, rmsd_matrix)
    print(">>> Hierarchical clustering (%s) complete " %linkage_type)
    return linkage_matrix

def runHierarchicalClustering(traj, args):
    '''
    DESCRIPTION
    Method for running Hierarchical clustering algorithm bases on type. Type
    refers to linkage factor. Filename and destintion are also used for input
    files and destination of desired output.

    Arguments:
        filename (str): string of filename.
        linkage_type (str): string for hierarchical type.
    '''
    traj = clean_trajectory(traj)
    rmsd_matrix_temp = preprocessing_hierarchical(traj)
    linkage_temp = cluserting(rmsd_matrix_temp, args.linkage)
    no_frames = getNumberOfFrames(traj)
    save_dendrogram(args.linkage, linkage_temp, True) # False not to show dendrogram
    produceClusters(linkage_temp, no_frames, args)

def randomDataValidation():
    '''
    DESCRIPTION
    Method for validation with random shapes of clusters for baseline test of algorithms.
    '''
    # Random seed
    numpy.random.seed(0)
    n_samples = 2000  # Sample size - 2000
    # Noisy cirle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
    # Noisy moon data
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    # Blob data
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    # Unstructured data
    no_structure = numpy.random.rand(n_samples, 2), None
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = numpy.dot(X, transformation)
    aniso = (X_aniso, y)
    # Blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

    X, y = aniso
    plot.scatter(X[:, 0], X[:, 1])
    plot.show()
    d = pdist(X, metric="euclidean")
    linkage_temp = cluserting(d, "ward")
    postprocessing.show_dendrogram("ward", linkage_temp)

def validation():
    # The iris dataset is available from the sci-kit learn package
    n_samples =2000
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    X,y = blobs
    d = pdist(X, metric="euclidean")

    iris = datasets.load_iris()
    type = "ward"
    # Plot results
    # print(iris.data[:, 0])

    # plot.scatter(iris.data[:, 3], iris.data[:, 0])
    # plot.xlabel('Petal width (cm)')
    # plot.ylabel('Sepal length (cm)')
    # plot.show()

    # plot.show()
    # Compute distance matrix
    # d = pdist(X=iris.data[:, [0, 3]], metric="euclidean")
    # print(numpy.ndim(d))
    reduced_distances = squareform(d, checks=True)
    # print(numpy.ndim(reduced_distances))
    # postprocessing.illustrateRMSD(reduced_distances)
    # Perform agglomerative hierarchical clustering
    # Use 'average' link function
    linkage_temp = cluserting(reduced_distances, type)
    no_frames = 2000
    save_dendrogram(type, linkage_temp, False)
    cophenetic(linkage_temp, reduced_distances)
    produceClusters(linkage_temp, no_frames, type)
    # plot.figure(figsize=(10, 8))
    #clusters = fcluster(linkage_temp, 3, criterion='maxclust')
    # plot.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    # plot.show()
    # Print the first 6 rows
    # Sepal Length, Sepal Width, Petal Length, Petal Width
    # print(iris.data)
    #

if __name__ == "__main__":
    # runHierarchicalClustering("MenY_aligned_downsamp10_reduced(Nic).pdb", "graphics", "ward")
    # validation()
    # randomDataValidation()
    runHierarchicalClustering("MenY_0_to_1000ns_aligned(100first).pdb", "ward")
    # validation()
