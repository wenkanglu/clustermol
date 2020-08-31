import argparse
import configparser
import copy
import os

# from params import Params
import select_algorithm
import select_preprocessing

os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # changes cwd to always be at clustermol
directory = os.getcwd()
# Params.cmd = directory

SUBPARSER_ARGS = "args"
SUBPARSER_CONF = "conf"
SUBPARSER_PREP = "prep"

algorithm_list = ["hdbscan", "hierarchical", "imwkmeans", "tsne", "umap"]
hierarchical_list = ["average", "complete", "single", "ward"]


def parse():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_used", help="Select a job")

    # subparser for handling config file/s
    parser_conf = subparsers.add_parser(SUBPARSER_CONF, help="Cluster using .ini configuration file/s")
    parser_conf.set_defaults(function=handle_configuration)
    parser_conf.add_argument("-c",
                             "--configuration",
                             default=None,
                             required=True,
                             help="Select configuration file/s", )

    # subparser for handling standard args
    parser_args = subparsers.add_parser(SUBPARSER_ARGS, help="Cluster using arguments")
    parser_args.set_defaults(function=call_algorithm)
    parser_args.add_argument("-a",
                             "--algorithm",
                             default=None,
                             required=True,
                             choices=algorithm_list,
                             help="Select a clustering algorithm", )
    parser_args.add_argument("-s",
                             "--source",
                             default=None,
                             required=True,
                             help="Select the input source", )
    parser_args.add_argument("-d",
                             "--destination",
                             default=None,
                             required=True,
                             help="Select output destination", )
    parser_args.add_argument("-v",
                             "--visualise",
                             default="false",
                             choices=["true", "false"],
                             help="Select whether to visualise cluster results", )
    parser_args.add_argument("-ds",
                             "--downsample",
                             default=None,
                             help="Select every nth frame to be kept", )
    parser_args.add_argument("-sel",
                             "--selection",
                             default=None,
                             help="Input a selection operation to be performed", )
    parser_args.add_argument("-sc",
                             "--saveclusters",
                             default=None,
                             help="Save the largest n number of clusters to destination", )
    # algorithm specific arguments
    parser_args.add_argument("-l",
                             "--linkage",
                             default=None,
                             choices=hierarchical_list,
                             help="Select linkage type if using hierarchical clustering", )

    # subparser for handling preprocessing jobs
    parser_prep = subparsers.add_parser(SUBPARSER_PREP, help="Perform a preprocessing job")
    parser_prep.set_defaults(function=call_preprocessing)
    parser_prep.add_argument("-s",
                             "--source",
                             dest="prep_source",
                             default=None,
                             required=True,
                             help="Select the file to be processed", )
    parser_prep.add_argument("-d",
                             "--destination",
                             dest="prep_destination",
                             default=None,
                             required=True,
                             help="Select destination of processed file", )
    parser_prep.add_argument("-ds",
                             "--downsample",
                             dest="prep_downsample",
                             default=None,
                             help="Select every nth frame to be kept", )
    parser_prep.add_argument("-sel",
                             "--selection",
                             dest="prep_selection",
                             default=None,
                             help="Input a selection operation to be performed", )

    args = parser.parse_args()
    # print(args)  # keep for debugging args
    args.function(args)


def handle_configuration(args):
    # config_path = args.configuration
    if os.path.isfile(args.configuration):
        parse_configuration(args, args.configuration)
    elif os.path.isdir(os.path.abspath(args.configuration)):
        for filename in os.listdir(args.configuration):
            parse_configuration(args, os.path.join(args.configuration, filename))


def parse_configuration(args, filename):
    # print("hello world!")
    if filename.endswith(".ini"):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        for section in config.sections():
            if section[0] == "c":
                # print(section)
                args_copy = copy.copy(args)
                # print(section)
                args_copy.algorithm = config[section]["--algorithm"]  # sets algorithm from config file
                # print(args_copy.algorithm)
                args_copy.source = config[section]["--source"]
                args_copy.destination = config[section]["--destination"]
                args_copy.visualise = config[section]["--visualise"]
                if config.has_option(section, "--downsample"):
                    args_copy.downsample = config[section]["--downsample"]
                if config.has_option(section, "--selection"):
                    args_copy.selection = config[section]["--selection"]
                if config.has_option(section, "--saveclusters"):
                    args_copy.saveclusters = config[section]["--saveclusters"]
                if args_copy.algorithm == "hierarchical":
                    args_copy.linkage = config[section]["--linkage"]
                call_algorithm(args_copy)
            elif section[0] == "p":
                args_copy.prep_source = config[section]["--source"]
                args_copy.prep_destination = config[section]["--destination"]
                if config.has_option(section, "--downsample"):
                    args_copy.prep_downsample = config[section]["--downsample"]
                if config.has_option(section, "--selection"):
                    args_copy.prep_selection = config[section]["--selection"]
    else:
        print(args.configuration + " is not .ini type")


def call_algorithm(args):
    # print(args)
    if args.visualise == "true":
        args.visualise = True
    else:
        args.visualise = False
    if args.algorithm == "hierarchical":
        # print(args.visualise)
        select_algorithm.call_hierarchical(args)
    # call algorithm with these args
    elif args.algorithm == "imwkmeans":
        select_algorithm.call_imwkmeans(args)
    elif args.algorithm == "hdbscan":
        select_algorithm.call_hdbscan(args)
    elif args.algorithm == "tsne":
        select_algorithm.call_tsne(args)
    elif args.algorithm == "umap":
        select_algorithm.call_umap(args)


def call_preprocessing(args):
    # TODO: call preprocessing under processing/pre when a caller method in select_preprocessing has been made
    None


if __name__ == "__main__":
    parse()
