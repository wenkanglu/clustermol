from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from constants import SILHOUETTE, DAVIESBOULDIN, CALINSKIHARABASZ

def handle_args(args):
    None

def label_counts(labels):
    unique, counts = numpy.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def calculate_CVI(indices, dest, data, labels):
    with open ('data/data_dest/' + dest + '_cviresults.txt', 'w') as f:
        if SILHOUETTE in indices: #sample-size!
            f.write("Silhouette score is {0}\n".format(silhouette_score(data, labels)))
        if DAVIESBOULDIN in indices:
            f.write("Davies-Bouldin score is {0}\n".format(davies_bouldin_score(data, labels)))
        if CALINSKIHARABASZ in indices:
            f.write("Calinski and Harabasz score is {0}\n".format(calinski_harabasz_score(data, labels)))

def save_largest_clusters(traj, n, labels):
    None
