import os

import mdtraj

from constants import SUBPARSER_CLUS, HIERARCHICAL, IMWKMEANS, HDBSCAN, TSNE, UMAP
from constants import SUBPARSER_PREP
from processing import pre_placeholder
from algorithms.hierarchical import hierarchical
from algorithms.imwkmeans import cluster_imwkmeans
from algorithms.hdbscan import hdbscan
from algorithms.tsne import tsne
from algorithms.umap import umap


def start_job(args, job):
    print("Loading trajectory from file...")
    traj = mdtraj.load(os.path.join("data", "data_src", args.source))
    if args.selection:
        sel = traj.topology.select(args.selection)
        traj = traj.atom_slice(sel)

    if args.downsample:
        traj = traj[::int(args.downsample)]

    if job == SUBPARSER_CLUS:
        if args.visualise == "true":
            args.visualise = True
        else:
            args.visualise = False

        if args.algorithm == HIERARCHICAL:
            hierarchical.runClustering(traj, args.source, args.destination, args.linkage, args.visualise)
        elif args.algorithm == IMWKMEANS:
            cluster_imwkmeans.cluster(traj, args)
        elif args.algorithm == HDBSCAN:
            hdbscan.cluster(traj, args)
        elif args.algorithm == TSNE:
            tsne.cluster(traj, args)
        elif args.algorithm == UMAP:
            umap.umap_main(traj, args)

    elif job == SUBPARSER_PREP:
        if args.destination.endswith(".pdb"):
            traj.save_pdb("data/data_src/" + args.destination)
        else:
            traj.save_pdb("data/data_src/" + args.destination + ".pdb")
