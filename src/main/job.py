import os
import copy
import numpy as np
from traceback import print_exception

import mdtraj
from processing import post_proc

from main.constants import SUBPARSER_CLUS, HIERARCHICAL, QT, QTVECTOR, IMWKMEANS, HDBSCAN, TSNE, UMAP, DATA_SRC, DATA_DEST, \
    DATA, IRIS, BREASTCANCER, DIGITS, WINE, TEST, NOISE, BLOBS, VBLOBS, LINKAGE, K_CLUSTERS, DDISTANCE, QUALITYTHRESHOLD, MINSAMPLES, MINCLUSTERSIZE, \
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
from sklearn import cluster, datasets, mixture

directory = os.getcwd()


def start_job(args, job):
    input_data = None
    traj_unselected = None
    if args.source:
        print("Loading trajectory from file...")
        try:
            input_data = mdtraj.load(os.path.join(directory + DATA, DATA_SRC, args.source))
        except IOError:
            print(args.source + " could not be found.")
        print("Trajectory load complete:")
        print(input_data)
        if args.frameselect or args.downsample:
            start = args.frameselect[0] if args.frameselect else None
            end = args.frameselect[1] if args.frameselect  else None
            step = args.downsample if args.downsample else None
            input_data = input_data[start:end:step]
            print("Trajectory frame selection and/or downsample complete:")
            print(input_data)

        if args.saveclusters:
            traj_unselected = copy.copy(input_data)

        if args.selection:
            try:
                sel = input_data.topology.select(args.selection)
                input_data = input_data.atom_slice(sel)
                print("Trajectory atom selection operation complete:")
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
        elif args.test == NOISE:
            noise = np.random.rand(60, 3), None
            input_data, y = noise
        elif args.test == BLOBS:
            blobs = datasets.make_blobs(n_samples=60, centers =6, random_state=3, cluster_std =0.2, center_box=(0, 10))
            input_data, y = blobs
        elif args.test == VBLOBS:
            cluster_std = [5.0, 2.5, 0.5, 1.0, 1.1, 0.0]
            varied_blobs = datasets.make_blobs(n_samples=60, centers =6, cluster_std=cluster_std, random_state=3, center_box=(0, 10))
            input_data, y = varied_blobs


    try:
        if job == SUBPARSER_CLUS:
            if args.visualise == "true":
                args.visualise = True
            else:
                args.visualise = False

            if not os.path.isdir(os.path.join(directory + DATA, DATA_DEST, args.destination)):
                os.mkdir(os.path.join(directory + DATA, DATA_DEST, args.destination))

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

            labels = None
            if args.algorithm == HIERARCHICAL:
                if args.linkage is None:
                    raise Exception(LINKAGE + " is required for " + HIERARCHICAL)
                if args.k_clusters is None and args.ddistance is None:
                    raise Exception("Either " + K_CLUSTERS + " or " + DDISTANCE + " is required for " + HIERARCHICAL)
                if args.k_clusters is not None and args.ddistance is not None:
                    raise Exception("Only " + K_CLUSTERS + " or " + DDISTANCE + " is required for " + HIERARCHICAL)
                labels = hierarchical.runHierarchicalClustering(input_data, args)
            elif args.algorithm == QT:
                if args.qualitythreshold is None:
                    raise Exception(QUALITYTHRESHOLD + " is required for " + QT)
                if args.minclustersize is None:
                    raise Exception(MINCLUSTERSIZE + " is required for " + QT)
                labels = qt.cluster(input_data, "qt_original", args)
            elif args.algorithm == QTVECTOR:
                if args.qualitythreshold is None:
                    raise Exception(QUALITYTHRESHOLD + " is required for " + QTVECTOR)
                if args.minclustersize is None:
                    raise Exception(MINCLUSTERSIZE + " is required for " + QTVECTOR)
                labels = qt.cluster(input_data, "qt_vector", args)
            elif args.algorithm == IMWKMEANS:
                labels = cluster_imwkmeans.cluster(input_data, args)
            elif args.algorithm == HDBSCAN:
                if args.minclustersize is None:
                    raise Exception(MINCLUSTERSIZE + " is required for " + HDBSCAN)
                if args.minsamples is None:
                    raise Exception(MINSAMPLES + " is required for " + HDBSCAN)
                labels = hdbscan.cluster(input_data, args)

            post_proc.saveClusters(labels, args.destination, args.algorithm) # save cluster text file

            if args.saveclusters:
                post_proc.save_largest_clusters(int(args.saveclusters), traj_unselected, labels, args.destination, args.algorithm)

            if args.visualise:
                post_proc.scatterplot_cluster(labels, args.destination, args.algorithm)
                # TODO: Open VMD and show cluster results here.

            #write results
            counts = post_proc.label_counts(labels, args.algorithm, args.destination) # must be run first to create file
            if args.validate:
                if len(counts)>1:
                    post_proc.calculate_CVI(args.validate, input_data, labels, args.destination, args.algorithm)
                else:
                    print("Error: CVIs can only be calculated when more than one cluster is produced.")

        elif job == SUBPARSER_PREP:
            if args.destination.endswith(".pdb"):
                input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination))
            else:
                input_data.save_pdb(os.path.join(DATA, DATA_SRC, args.destination + ".pdb"))

    except IOError:
        print("Error in handling " + args.destination + ". Check if data/data_dest/<subfolders>/ exist.")
