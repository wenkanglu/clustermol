import argparse
import configparser
import copy
import os
import sys

from main.constants import SUBPARSER_CONF, SUBPARSER_CLUS, SUBPARSER_PREP, UMAP, TSNE, IMWKMEANS, HIERARCHICAL, HDBSCAN, \
    ALGORITHM, SOURCE, DESTINATION, VISUALISE, VALIDATE, DOWNSAMPLE, FRAMESELECT, SELECTION, SAVECLUSTERS, LINKAGE, \
    MINCLUSTERSIZE, MINSAMPLES, CONFIGURATION, AVERAGE, COMPLETE, SINGLE, WARD, SILHOUETTE, DAVIESBOULDIN, \
    CALINSKIHARABASZ, QT, QTVECTOR, \
    QUALITYTHRESHOLD, K_CLUSTERS, DDISTANCE, CONFIGS, DATA, N_COMPONENTS, N_NEIGHBOURS, PREPROCESS, \
    IRIS, DIGITS, WINE, BREASTCANCER, TEST, NOISE, BLOBS, VBLOBS, DATA_DEST, DATA_SRC

from main.job import start_job

sys.setrecursionlimit(10**6)

algorithm_list = [HDBSCAN, HIERARCHICAL, IMWKMEANS, QT, QTVECTOR]
hierarchical_list = [AVERAGE, COMPLETE, SINGLE, WARD]
validity_indices = [CALINSKIHARABASZ, DAVIESBOULDIN, SILHOUETTE]
preprocess_list = [TSNE, UMAP]
test_data_list = [IRIS, BREASTCANCER, DIGITS, WINE, NOISE, BLOBS, VBLOBS]


def parse():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_used", help="Select a job")

    # subparser for handling config file/s
    parser_conf = subparsers.add_parser(SUBPARSER_CONF, help="Cluster using .ini configuration file/s")
    parser_conf.set_defaults(function=handle_configuration)
    parser_conf.add_argument("-c",
                             CONFIGURATION,
                             default=None,
                             required=True,
                             help="Select configuration file/s", )

    # subparser for handling standard args
    parser_clus = subparsers.add_parser(SUBPARSER_CLUS, help="Cluster using arguments")
    parser_clus.set_defaults(function=start_job)
    parser_clus.add_argument("-a",
                             ALGORITHM,
                             default=None,
                             required=True,
                             choices=algorithm_list,
                             help="Select a clustering algorithm", )
    parser_clus.add_argument("-s",
                             SOURCE,
                             default=None,
                             help="Select the input source", )
    parser_clus.add_argument("-d",
                             DESTINATION,
                             default=None,
                             required=True,
                             help="Select output destination", )
    parser_clus.add_argument("-v",
                             VISUALISE,
                             default="false",
                             choices=["true", "false"],
                             help="Select whether to visualise cluster results", )
    parser_clus.add_argument("-cvi",
                             VALIDATE,
                             default=None,
                             nargs='+',
                             choices=validity_indices,
                             help="Select CVIs to calculate from cluster results.", )
    parser_clus.add_argument("-ds",
                             DOWNSAMPLE,
                             default=None,
                             type=int,
                             help="Select every nth frame to be kept", )
    parser_clus.add_argument("-fs",
                             FRAMESELECT,
                             default=None,
                             nargs=2,
                             type=int,
                             help="Select start and end of frames to cluster on", )
    parser_clus.add_argument("-sel",
                             SELECTION,
                             default=None,
                             help="Input a selection operation to be performed", )
    parser_clus.add_argument("-sc",
                             SAVECLUSTERS,
                             default=None,
                             type=int,
                             help="Save the largest n number of clusters to destination", )
    parser_clus.add_argument("-p",
                             PREPROCESS,
                             default=None,
                             choices=preprocess_list,
                             help="Select a preprocessing technique", )
    # algorithm specific arguments
    parser_clus.add_argument("-l",
                             LINKAGE,
                             default=None,
                             choices=hierarchical_list,
                             help="Select linkage type if using hierarchical clustering", )
    parser_clus.add_argument("-mc",
                             MINCLUSTERSIZE,
                             default=None,
                             type=int,
                             help="Minimum cluster size for HDBSCAN or Quality Threshold clustering", )
    parser_clus.add_argument("-ms",
                             MINSAMPLES,
                             default=None,
                             type=int,
                             help="Minimum samples for HDBSCAN clustering", )
    parser_clus.add_argument("-qt",
                             QUALITYTHRESHOLD,
                             default=None,
                             type=float,
                             help="Minimum Cutoff/Quality Threshold value for Quality Threshold clustering", )
    parser_clus.add_argument("-k",
                             K_CLUSTERS,
                             default=None,
                             type=int,
                             help="Number of assumed clusters for Hierarchical clustering", )
    parser_clus.add_argument("-dist",
                             DDISTANCE,
                             default=None,
                             type=float,
                             help="Distance cutoff for Hierarchical clustering mergers", )
    parser_clus.add_argument("-nc",
                             N_COMPONENTS,
                             default=None,
                             type=int,
                             help="Number of components (dimensions) for UMAP or t-SNE to embed to")
    parser_clus.add_argument("-nn",
                             N_NEIGHBOURS,
                             default=None,
                             type=int,
                             help="Number of neighbours for UMAP or perplexity for t-SNE, higher values focus on "
                                  "global structure")
    parser_clus.add_argument("-t",
                             TEST,
                             default=None,
                             choices=test_data_list,
                             help="Select test data as source", )

    # subparser for handling preprocessing jobs
    parser_prep = subparsers.add_parser(SUBPARSER_PREP, help="Perform a preprocessing job")
    parser_prep.set_defaults(function=start_job)
    parser_prep.add_argument("-s",
                             SOURCE,
                             default=None,
                             required=True,
                             help="Select the file to be processed", )
    parser_prep.add_argument("-d",
                             DESTINATION,
                             default=None,
                             required=True,
                             help="Select destination of processed file", )
    parser_prep.add_argument("-ds",
                             DOWNSAMPLE,
                             default=None,
                             type=int,
                             help="Select every nth frame to be kept", )
    parser_prep.add_argument("-fs",
                             FRAMESELECT,
                             default=None,
                             nargs='*',
                             type=int,
                             help="Select start and end of frames to keep", )
    parser_prep.add_argument("-sel",
                             SELECTION,
                             default=None,
                             help="Input a selection operation to be performed", )

    args = parser.parse_args()
    # print(args)  # keep for debugging args
    if args.subparser_used == SUBPARSER_CONF:
        args.function(args)
    else:
        args.function(args, args.subparser_used)


def handle_configuration(args):
    print(DATA + CONFIGS + args.configuration)
    if os.path.isfile(DATA + CONFIGS + args.configuration):
        parse_configuration(args, DATA + CONFIGS + args.configuration)
    elif os.path.isdir(os.path.abspath(DATA + CONFIGS + args.configuration)):
        for filename in os.listdir(DATA + CONFIGS + args.configuration):
            parse_configuration(args, os.path.join(DATA + CONFIGS + args.configuration, filename))
    else:
        print("Error - Cannot find config file.")


def parse_configuration(args, filename):
    current = None
    if filename.endswith(".ini"):
        try:
            config = configparser.ConfigParser(allow_no_value=False)
            config.read(filename)
            for section in config.sections():
                print("Working on section " + section)
                # if section is for clustering job
                try:
                    if section[0] == "c":
                        # general required arguments
                        args_copy = copy.copy(args)
                        current = ALGORITHM
                        if config.has_option(section, ALGORITHM):
                            args_copy.algorithm = config[section][ALGORITHM]  # sets algorithm from config file
                            if args_copy.algorithm not in algorithm_list:
                                raise Exception(
                                    args_copy.algorithm + " is not a valid option."
                                )
                        else:
                            raise KeyError
                        current = SOURCE + "/" + TEST
                        if config.has_option(section, SOURCE):
                            args_copy.source = config[section][SOURCE]
                            args_copy.test = None
                        elif config.has_option(section, TEST):
                            args_copy.test = config[section][TEST]
                            if args_copy.test not in test_data_list:
                                raise Exception(
                                    args_copy.test + " is not a valid option."
                                )
                            args_copy.source = None
                        else:
                            raise KeyError
                        current = DESTINATION
                        args_copy.destination = config[section][DESTINATION]

                        if config.has_option(section, VISUALISE):
                            args_copy.visualise = config[section][VISUALISE]
                        else:
                            args_copy.visualise = "false"
                        # trajectory preprocessing
                        if config.has_option(section, VALIDATE):
                            args_copy.validate = config[section][VALIDATE].split()
                            for cvi in args_copy.validate:
                                if cvi not in validity_indices:
                                    raise Exception(
                                        cvi + " is not a valid option."
                                    )
                        else:
                            args_copy.validate = None
                        if config.has_option(section, DOWNSAMPLE):
                            args_copy.downsample = int(config[section][DOWNSAMPLE])
                        else:
                            args_copy.downsample = None
                        if config.has_option(section, FRAMESELECT):
                            fs = config[section][FRAMESELECT].split()
                            for i in range(len(fs)):
                                fs[i] = int(fs[i])
                            args_copy.frameselect = fs
                        else:
                            args_copy.frameselect = None
                        if config.has_option(section, SELECTION):
                            args_copy.selection = config[section][SELECTION]
                        else:
                            args_copy.selection = None
                        if config.has_option(section, SAVECLUSTERS):
                            args_copy.saveclusters = int(config[section][SAVECLUSTERS])
                        else:
                            args_copy.saveclusters = None
                        # cluster preprocessing
                        if config.has_option(section, PREPROCESS):
                            args_copy.preprocess = config[section][PREPROCESS]
                            if args_copy.preprocess == UMAP or args_copy.preprocess == TSNE:
                                current = N_COMPONENTS
                                args_copy.ncomponents = int(config[section][N_COMPONENTS])
                                current = N_NEIGHBOURS
                                args_copy.nneighbours = int(config[section][N_NEIGHBOURS])
                            else:
                                raise Exception(
                                    args_copy.preprocess + " is not a valid option"
                                )
                        else:
                            args_copy.preprocess = None
                        # hierarchical
                        if args_copy.algorithm == HIERARCHICAL:
                            current = LINKAGE
                            args_copy.linkage = config[section][LINKAGE]
                            if args_copy.linkage not in hierarchical_list:
                                raise Exception(
                                    args_copy.linkage + " is not a valid option."
                                )
                            current = K_CLUSTERS + "/" + DDISTANCE
                            if config.has_option(section, K_CLUSTERS):
                                args_copy.k_clusters = config[section][K_CLUSTERS]
                                args_copy.ddistance = None
                            elif config.has_option(section, DDISTANCE):
                                args_copy.ddistance = config[section][DDISTANCE]
                                args_copy.k_clusters = None
                            else:
                                raise KeyError
                        # hdbscan
                        if args_copy.algorithm == HDBSCAN:
                            current = MINCLUSTERSIZE
                            args_copy.minclustersize = int(config[section][MINCLUSTERSIZE])
                            current = MINSAMPLES
                            args_copy.minsamples = int(config[section][MINSAMPLES])
                        # qt
                        if args_copy.algorithm == QT or args_copy.algorithm == QTVECTOR:
                            current = QUALITYTHRESHOLD
                            args_copy.qualitythreshold = float(config[section][QUALITYTHRESHOLD])
                            current = MINCLUSTERSIZE
                            args_copy.minclustersize = int(config[section][MINCLUSTERSIZE])
                        start_job(args_copy, SUBPARSER_CLUS)
                    # if section is for preprocessing job
                    elif section[0] == "p":
                        args_copy = copy.copy(args)
                        current = SOURCE
                        args_copy.source = config[section][SOURCE]
                        current = DESTINATION
                        args_copy.destination = config[section][DESTINATION]
                        if config.has_option(section, DOWNSAMPLE):
                            args_copy.downsample = int(config[section][DOWNSAMPLE])
                        else:
                            args_copy.downsample = None
                        if config.has_option(section, FRAMESELECT):
                            fs = config[section][FRAMESELECT].split()
                            for i in range(len(fs)):
                                fs[i] = int(fs[i])
                            args_copy.frameselect = fs
                        else:
                            args_copy.frameselect = None
                        if config.has_option(section, SELECTION):
                            args_copy.selection = config[section][SELECTION]
                        else:
                            args_copy.selection = None
                        start_job(args_copy, SUBPARSER_PREP)

                    else:
                        print("Config sections must start with 'p' or 'c' for processing, and clustering jobs "
                              "respectively.")
                except ValueError:
                    print("Error in string-to-int/float conversion in section " + section +
                          ".\nPlease ensure that the expected numeric parameters are represented as such.")
                except KeyError:
                    print("Required parameter " + current + " could not be found in section " + section + ".")
        except configparser.ParsingError:
            print("Interpolation of a parameter has failed.\nPlease ensure that each option has an associated value.")
    else:
        print(args.configuration + " is not .ini type. Skipped.")


def create_dirs():
    if not os.path.isdir(DATA):
        os.mkdir(DATA)
        os.mkdir(os.path.join(DATA, DATA_DEST))
        os.mkdir(os.path.join(DATA, DATA_SRC))
        os.mkdir(os.path.join(DATA, CONFIGS))
    else:
        if not os.path.isdir(os.path.join(DATA, DATA_SRC)):
            os.mkdir(os.path.join(DATA, DATA_SRC))
        if not os.path.isdir((os.path.join(DATA, DATA_DEST))):
            os.mkdir(os.path.join(DATA, DATA_DEST))
        if not os.path.isdir((os.path.join(DATA, CONFIGS))):
            os.mkdir(os.path.join(DATA, CONFIGS))


def main():
    create_dirs()
    parse()
