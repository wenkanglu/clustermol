from algorithms.hierarchical import hierarchical
from algorithms.imwkmeans import cluster_imwkmeans
from algorithms.tsne import tsne


def call_hierarchical(args):
    hierarchical.runClustering(args.source, args.destination, args.linkage, args.visualise)


def call_imwkmeans(args):
    cluster_imwkmeans.cluster(args)


def call_hdbscan(args):
    cluster_imwkmeans.cluster_hdbscan(args)


def call_tsne(args):
    tsne.cluster(args)
