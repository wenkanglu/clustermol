import os
import copy

import mdtraj
from processing import post_proc

from main.constants import SUBPARSER_CLUS, HIERARCHICAL, QT, QTVECTOR, IMWKMEANS, HDBSCAN, TSNE, UMAP, DATA_SRC, DATA, \
    IRIS, BREASTCANCER, DIGITS, WINE
from main.constants import SUBPARSER_PREP
from algorithms.hierarchical import hierarchical
from algorithms.qt import qt
from algorithms.imwkmeans import cluster_imwkmeans
from algorithms.hdbscan import hdbscan
from algorithms.pca import pca_script
from algorithms.tsne import tsne_script
from algorithms.umap import umap_script

from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer


def start_job(args, job):
    input_data = None
    if args.source:
        print("Loading trajectory from file...")
        input_data = mdtraj.load(os.path.join(DATA, DATA_SRC, args.source))
        if args.selection:
            sel = input_data.topology.select(args.selection)
            input_data = input_data.atom_slice(sel)

        if args.downsample:
            input_data = input_data[::int(args.downsample)]
    elif args.test:
        if args.test == IRIS:
            input_data = load_iris().data
        elif args.test == BREASTCANCER:
            input_data = load_breast_cancer().data
        elif args.test == DIGITS:
            input_data = load_digits().data
        elif args.test == WINE:
            input_data = load_wine().data

    if args.preprocess:
        if args.preprocess == TSNE:
            input_data = tsne_script.tsne_main(pca_script.pca_main(input_data, args), args)
        elif args.preprocess == UMAP:
            input_data = umap_script.umap_main(input_data, args)

    if job == SUBPARSER_CLUS:
        if args.visualise == "true":
            args.visualise = True
        else:
            args.visualise = False

        labels = None
        if args.algorithm == HIERARCHICAL:
            hierarchical.runHierarchicalClustering(input_data, args)
        elif args.algorithm == QT:
            qt.cluster(input_data, "qt_original", args)
        elif args.algorithm == QTVECTOR:
            qt.cluster(input_data, "qt_vector", args)
        elif args.algorithm == IMWKMEANS:
            labels = cluster_imwkmeans.cluster(traj, args)
        elif args.algorithm == HDBSCAN:
            labels = hdbscan.cluster(traj, args)

        if args.saveclusters:
             post_proc.save_largest_clusters(int(args.saveclusters), traj_unselected, labels, args.destination)

    elif job == SUBPARSER_PREP:
        if args.destination.endswith(".pdb"):
            input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination))
        else:
            input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination + ".pdb"))
