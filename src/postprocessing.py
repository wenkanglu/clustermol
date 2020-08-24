import os
import numpy
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy


def scatterplot_single(clusters_arr, no_frames, qt_type):
        '''
        DESCRIPTION
        Produce cluster scatter plot of frames

        Arguments:
            clusters_arr (numpy.ndarray): cluster indexes per frame.
            no_frames (int): number of frames
            qt_type (str): qt types of implementation.
        '''
        plot.figure()
        plot.scatter(numpy.arange(no_frames), clusters_arr, marker = '+')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Number")
        plot.title("Scatter Plot - %s" %qt_type )
        os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig("Scatterplot_%s.png" %qt_type)
        plot.close()

def scatterplot_multiple(clusters_arr1, cluster_arr2, no_frames):
        '''
        DESCRIPTION
        Produce cluster scatter plot of frames

        Arguments:
            clusters_arr1 (numpy.ndarray): cluster indexes per frame.
            clusters_arr2 (numpy.ndarray): cluster indexes per frame.
            no_frames (int): number of frames
            qt_type1 (str): qt types of implementation.
            qt_type2 (str): qt types of implementation.
        '''
        plot.figure()
        print(cluster_arr2)
        plot.scatter(numpy.arange(no_frames), clusters_arr1, marker = '+', color='blue')
        plot.scatter(numpy.arange(no_frames), cluster_arr2, marker = '+', color='red')
        plot.xlabel("Frame Number")
        plot.ylabel("Cluster Number")
        plot.title("Scatter Plot - qt_orginal and qt_like")
        os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
        #print(os.getcwd())
        plot.savefig("Scatterplot_comparison.png")
        plot.close()

def saveClusters(clusters_arr, qt_type):
    '''
    DESCRIPTION
    Save cluster indexes to text file

    Arguments:
        clusters_arr (numpy.ndarray): Cluster indexes per frame.
        qt_type (str): qt types of implementation.
    '''
    os.chdir(os.path.join(os.path.dirname(__file__), '..')+ "/data/data_dest/")
    numpy.savetxt("Clusters_%s.txt" %qt_type, clusters_arr, fmt='%i')

def show_dendrogram(hierarchical_type, linkage):
    '''
    DESCRIPTION
    Popup Dendrogram produced by hierarchical clustering.

    Arguments:
        hierarchical_type (str): string for hierarchical type.
        linkage (numpy.ndarray): linkage matrix from clustering.
    '''
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage.astype("float64"), no_labels=True)
    plot.show()

def save_dendrogram(hierarchical_type, linkage, destination):
    '''
    DESCRIPTION
    Save Dendrogram produced by hierarchical clustering.

    Arguments:
        rmsd_matrix_temp (numpy.ndarray): rmsd matrix used for clustering.
        hierarchical_type (str): string for hierarchical type.
        destination (str): string for file location within data_dest folder
    '''
    os.chdir(os.path.join(os.path.dirname(__file__), "..")+ "/data/data_dest/")  # changes cwd to always be at clustermol
    plot.title('RMSD %s linkage hierarchical clustering' %hierarchical_type)
    _ = scipy.cluster.hierarchy.dendrogram(linkage, no_labels=True)
    plot.savefig("dendrogram-clustering-%s.png" % hierarchical_type)



if __name__ == "__main__":
    print("Output Class")
