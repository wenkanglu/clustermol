from algorithms.hierarchical import hierarchical
from algorithms.deAmorim import clusterer
from algorithms.tsne import tsne


def call_hierarchical(args):
    hierarchical.runClustering(args.source, args.destination, args.linkage, args.visualise)


def call_imwkmeans(args):
    clusterer.cluster_imwkmeans(args)


def call_tsne(args):
    tsne.cluster(args)
