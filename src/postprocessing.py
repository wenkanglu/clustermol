import os
import numpy
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy


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
        os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig("scatterplot-%s.png" %cluster_type)
        plot.show()
        plot.close()


def scatterplot_multiple(clusters_arr1, cluster_arr2, no_frames):
        '''
        DESCRIPTION
        Produce cluster scatter plot of frames

        Arguments:
            clusters_arr1 (numpy.ndarray): cluster indexes per frame.
            clusters_arr2 (numpy.ndarray): cluster indexes per frame.
            no_frames (int): number of frames
        '''
        plot.figure()
        plot.scatter(numpy.arange(no_frames), clusters_arr1, marker = '^', color='blue',label='qt_like')
        plot.scatter(numpy.arange(no_frames), cluster_arr2, marker = 'v', color='red', label='qt_orginal')
        plot.legend(loc='uploter left')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Number")
        plot.title("Scatter Plot - qt_orginal and qt_like")
        os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig("scatterplot-comparison.png")

def saveClusters(clusters_arr, type):
    '''
    DESCRIPTION
    Save cluster indexes to text file

    Arguments:
        clusters_arr (numpy.ndarray): Cluster indexes per frame.
        qt_type (str): qt types of implementation.
    '''
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    numpy.savetxt("clusters-%s.txt" %type, clusters_arr, fmt='%i')

def save_dendrogram(hierarchical_type, linkage, flag_display):
    '''
    DESCRIPTION
    Popup Dendrogram produced by hierarchical clustering.

    Arguments:
        hierarchical_type (str): string for hierarchical type.
        linkage (numpy.ndarray): linkage matrix from clustering.
    '''
    os.chdir(os.path.join(os.path.dirname(__file__), "..")+ "/data/data_dest/")
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

def illustrateRMSD(rmsd_matrix):
    '''
    DESCRIPTION
    Creates an RMDS heatmap

    Arguments:
        rmsd_matrix (numpy.ndarray): rmsd matrix.
    '''
    plot.figure()
    plot.imshow(rmsd_matrix, cmap='viridis', interpolation='nearest')
    print(">>> Max pairwise rmsd: %f nm" % numpy.max(rmsd_matrix))
    print(">>> Average pairwise rmsd: %f nm" % numpy.mean(rmsd_matrix))
    print(">>> Median pairwise rmsd: %f nm" % numpy.median(rmsd_matrix))
    plot.colorbar()
    plot.savefig("RMSD-matrix.png")
    plot.show()
    plot.close()


def rmsd_vs_frame(no_frames, rmsds):
    '''
    DESCRIPTION
    Produce cluster scatter plot of frames. Skips first frame that computed against.

    Arguments:
        clusters_arr (mdtraj.traj): trajectory.
    '''
    plot.figure()
    plot.plot(numpy.arange(1,no_frames), rmsds[1:, ], 'r', label='all atoms')
    plot.legend()
    plot.title('RMSDs over time agaist first frame')
    plot.xlabel('Simulation frames')
    plot.ylabel('RMSD (nm)')
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    plot.savefig("rmsd-vs-frame.png")
    plot.show()
    plot.close()

if __name__ == "__main__":
    print("Output Class")
    # illustrateRMSD()
