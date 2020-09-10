import os
import copy

import mdtraj
from processing import post_proc

from main.constants import SUBPARSER_CLUS, HIERARCHICAL, QT, QTVECTOR, IMWKMEANS, HDBSCAN, TSNE, UMAP, DATA_SRC, DATA, \
    IRIS, BREASTCANCER, DIGITS, WINE, LINKAGE, K_CLUSTERS, DDISTANCE, QUALITYTHRESHOLD, MINSAMPLES, MINCLUSTERSIZE, \
    N_NEIGHBOURS, N_COMPONENTS
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
    traj_unselected = None
    if args.source:
        print("Loading trajectory from file...")
        try:
            input_data = mdtraj.load(os.path.join(DATA, DATA_SRC, args.source))
        except IOError:
            print(args.source + " could not be found.")
        print("Trajectory load complete:")
        print(input_data)
        if args.downsample:
            input_data = input_data[::int(args.downsample)]
            print("Trajectory downsample complete:")
            print(input_data)

        # TODO if start/end selection

        if args.saveclusters:
            traj_unselected = copy.copy(input_data)

        if args.selection:
            try:
                sel = input_data.topology.select(args.selection)
                input_data = input_data.atom_slice(sel)
                print("Trajectory selection operation complete:")
                print(input_data)
            except:
                print("Selection statement could not be parsed. Please check if it correct.")

    elif args.test:
        if args.test == IRIS:
            input_data = load_iris().data
        elif args.test == BREASTCANCER:
            input_data = load_breast_cancer().data
        elif args.test == DIGITS:
            input_data = load_digits().data
        elif args.test == WINE:
            input_data = load_wine().data

    if job == SUBPARSER_CLUS:
        if args.visualise == "true":
            args.visualise = True
        else:
            args.visualise = False

        if args.preprocess:
            if args.preprocess == TSNE:
                if args.nneighbours is None:
                    raise Exception(N_NEIGHBOURS + " is required for " + TSNE)
                if args.ncomponents is None:
                    raise Exception(N_COMPONENTS + " is required for " + TSNE)
                input_data = tsne_script.tsne_main(pca_script.pca_main(input_data, args), args)
            elif args.preprocess == UMAP:
                if args.nneighbours is None:
                    raise Exception(N_NEIGHBOURS + " is required for " + UMAP)
                if args.ncomponents is None:
                    raise Exception(N_COMPONENTS + " is required for " + UMAP)
                input_data = umap_script.umap_main(input_data, args)

        try:
            labels = None
            if args.algorithm == HIERARCHICAL:
                if args.linkage is None:
                    raise Exception(LINKAGE + " is required for " + HIERARCHICAL)
                if args.numberofclusters is None and args.dendrogramdistance is None:
                    raise Exception("Either " + K_CLUSTERS + " or " + DDISTANCE + " is required for " + HIERARCHICAL)
                if args.numberofclusters is not None and args.dendrogramdistance is not None:
                    raise Exception("Only " + K_CLUSTERS + " or " + DDISTANCE + " is required for " + HIERARCHICAL)
                hierarchical.runHierarchicalClustering(input_data, args)
            elif args.algorithm == QT:
                if args.qualitythreshold is None:
                    raise Exception(QUALITYTHRESHOLD + " is required for " + QT)
                if args.minsamples is None:
                    raise Exception(MINSAMPLES + " is required for " + QT)
                qt.cluster(input_data, "qt_original", args)
            elif args.algorithm == QTVECTOR:
                if args.qualitythreshold is None:
                    raise Exception(QUALITYTHRESHOLD + " is required for " + QTVECTOR)
                if args.minsamples is None:
                    raise Exception(MINSAMPLES + " is required for " + QTVECTOR)
                qt.cluster(input_data, "qt_vector", args)
            elif args.algorithm == IMWKMEANS:
                labels = cluster_imwkmeans.cluster(input_data, args)
            elif args.algorithm == HDBSCAN:
                if args.minclustersize is None:
                    raise Exception(MINCLUSTERSIZE + " is required for " + HDBSCAN)
                if args.minsamples is None:
                    raise Exception(MINSAMPLES + " is required for " + HDBSCAN)
                labels = hdbscan.cluster(input_data, args)

            if args.saveclusters:
                 post_proc.save_largest_clusters(int(args.saveclusters), traj_unselected, labels, args.destination)

            elif job == SUBPARSER_PREP:
                if args.destination.endswith(".pdb"):
                    input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination))
                else:
                    input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination + ".pdb"))

        except IOError:
            print("Error in handling " + args.destination + ". Check if data/data_dest/<subfolders>/ exist.")

