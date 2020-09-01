import argparse
import configparser
import copy
import os

from constants import SUBPARSER_CONF, SUBPARSER_CLUS, SUBPARSER_PREP, UMAP, TSNE, IMWKMEANS, HIERARCHICAL, HDBSCAN, \
    ALGORITHM, SOURCE, DESTINATION, VISUALISE, DOWNSAMPLE, SELECTION, SAVECLUSTERS, LINKAGE, CONFIGURATION, AVERAGE, \
    COMPLETE, SINGLE, WARD
from job import start_job

os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # changes cwd to always be at clustermol
directory = os.getcwd()

algorithm_list = [HDBSCAN, HIERARCHICAL, IMWKMEANS, TSNE, UMAP]
hierarchical_list = [AVERAGE, COMPLETE, SINGLE, WARD]


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
                             required=True,
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
    parser_clus.add_argument("-ds",
                             DOWNSAMPLE,
                             default=None,
                             help="Select every nth frame to be kept", )
    parser_clus.add_argument("-sel",
                             SELECTION,
                             default=None,
                             help="Input a selection operation to be performed", )
    parser_clus.add_argument("-sc",
                             SAVECLUSTERS,
                             default=None,
                             help="Save the largest n number of clusters to destination", )

    # algorithm specific arguments
    parser_clus.add_argument("-l",
                             LINKAGE,
                             default=None,
                             choices=hierarchical_list,
                             help="Select linkage type if using hierarchical clustering", )

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
                             help="Select every nth frame to be kept", )
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
    if os.path.isfile(args.configuration):
        parse_configuration(args, args.configuration)
    elif os.path.isdir(os.path.abspath(args.configuration)):
        for filename in os.listdir(args.configuration):
            parse_configuration(args, os.path.join(args.configuration, filename))


def parse_configuration(args, filename):
    if filename.endswith(".ini"):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        for section in config.sections():
            if section[0] == "c":
                # print(section)
                args_copy = copy.copy(args)
                args_copy.algorithm = config[section][ALGORITHM]  # sets algorithm from config file
                args_copy.source = config[section][SOURCE]
                args_copy.destination = config[section][DESTINATION]
                args_copy.visualise = config[section][VISUALISE]
                if config.has_option(section, DOWNSAMPLE):
                    args_copy.downsample = config[section][DOWNSAMPLE]
                else:
                    args_copy.downsample = None
                if config.has_option(section, SELECTION):
                    args_copy.selection = config[section][SELECTION]
                else:
                    args_copy.selection = None
                if config.has_option(section, SAVECLUSTERS):
                    args_copy.saveclusters = config[section][SAVECLUSTERS]
                else:
                    args_copy.saveclusters = None
                if args_copy.algorithm == HIERARCHICAL:
                    args_copy.linkage = config[section][LINKAGE]
                else:
                    args_copy.linkage = None
                start_job(args_copy, SUBPARSER_CLUS)
            elif section[0] == "p":
                args_copy.source = config[section][SOURCE]
                args_copy.destination = config[section][DESTINATION]
                if config.has_option(section, DOWNSAMPLE):
                    args_copy.downsample = config[section][DOWNSAMPLE]
                else:
                    args_copy.downsample = None
                if config.has_option(section, SELECTION):
                    args_copy.selection = config[section][SELECTION]
                else:
                    args_copy.selection = None
                start_job(args_copy, SUBPARSER_PREP)
            else:
                print("Config sections must start with 'p' or 'c' for processing and clustering jobs respectively")
    else:
        print(args.configuration + " is not .ini type")


if __name__ == "__main__":
    parse()
