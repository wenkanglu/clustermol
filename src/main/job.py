import os

import mdtraj

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
    if args.selection:
        sel = traj.topology.select(args.selection)
        traj = traj.atom_slice(sel)

    if args.downsample:
        traj = traj[::int(args.downsample)]

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

        if args.algorithm == HIERARCHICAL:
            hierarchical.runHierarchicalClustering(traj, args)
        elif args.algorithm == QT:
            qt.cluster(traj, "qt_original", args)
        elif args.algorithm == QTVECTOR:
            qt.cluster(traj, "qt_vector", args)
        elif args.algorithm == IMWKMEANS:
            cluster_imwkmeans.cluster(traj, args)
        elif args.algorithm == HDBSCAN:
            hdbscan.cluster(traj, args)

    elif job == SUBPARSER_PREP:
        if args.destination.endswith(".pdb"):
            traj.save_pdb(os.path.join(DATA, DATA_SRC, args.destination))
        else:
            traj.save_pdb(os.path.join(DATA, DATA_SRC, args.destination + ".pdb"))
