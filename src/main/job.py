import os
import copy

import mdtraj
from processing import post_proc

from main.constants import SUBPARSER_CLUS, HIERARCHICAL, QT, QTVECTOR, IMWKMEANS, HDBSCAN, TSNE, UMAP, DATA_SRC, DATA
from main.constants import SUBPARSER_PREP
from algorithms.hierarchical import hierarchical
from algorithms.qt import qt
from algorithms.imwkmeans import cluster_imwkmeans
from algorithms.hdbscan import hdbscan
from algorithms.pca import pca_script
from algorithms.tsne import tsne_script
from algorithms.umap import umap_script


def start_job(args, job):
    print("Loading trajectory from file...")
    traj = mdtraj.load(os.path.join(DATA, DATA_SRC, args.source))
    traj_unselected = None
    print("Trajectory load complete:")
    print(traj)
    if args.downsample:
        traj = traj[::int(args.downsample)]
        print("Trajectory downsample complete:")
        print(traj)

    #TODO if start/end selection

    if args.saveclusters:
        traj_unselected = copy(traj)

    if args.selection:
        sel = traj.topology.select(args.selection)
        traj = traj.atom_slice(sel)
        print("Trajectory selection operation complete:")
        print(traj)

    if args.preprocess:
        if args.preprocess == TSNE:
            traj = tsne_script.tsne_main(pca_script.pca_main(traj, args), args)
        elif args.preprocess == UMAP:
            traj = umap_script.umap_main(traj, args)

    if job == SUBPARSER_CLUS:
        if args.visualise == "true":
            args.visualise = True
        else:
            args.visualise = False
            
        labels = None
        if args.algorithm == HIERARCHICAL:
            hierarchical.runHierarchicalClustering(traj, args)
        elif args.algorithm == QT:
            qt.cluster(traj, "qt_original", args)
        elif args.algorithm == QTVECTOR:
            qt.cluster(traj, "qt_vector", args)
        elif args.algorithm == IMWKMEANS:
            labels = cluster_imwkmeans.cluster(traj, args)
        elif args.algorithm == HDBSCAN:
            labels = hdbscan.cluster(traj, args)

        if args.saveclusters:
             post_proc.save_largest_clusters(int(args.saveclusters), traj_unselected, labels, args.destination)

    elif job == SUBPARSER_PREP:
        if args.destination.endswith(".pdb"):
            traj.save_pdb(os.path.join(DATA, DATA_SRC, args.destination))
        else:
            traj.save_pdb(os.path.join(DATA, DATA_SRC, args.destination + ".pdb"))
