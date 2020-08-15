import argparse
import configparser
import copy
import os
import hierarchical

SUBPARSER_CONF = "conf"
SUBPARSER_ARGS = "args"

os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # changes cwd to always be at clustermol
# print(os.getcwd())
algorithm_list = ["hierarchical", "imwkmeans"]
hierarchical_list = ["single", "average", "ward"]


def parse():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_used", help="Select one clustering method")

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
                             default=None,  # change
                             required=True,
                             help="Select the input source", )
    parser_args.add_argument("-d",
                             "--destination",
                             default=None,  # change
                             required=True,
                             help="Select output destination", )
    parser_args.add_argument("-v",
                             "--visualise",
                             default=False,
                             choices=[True, False],
                             type=bool,
                             help="Select whether to visualise cluster results", )
    # algorithm specific arguments
    parser_args.add_argument("-l",
                             "--linkage",
                             default=None,
                             choices=hierarchical_list,
                             help="Select linkage type if using hierarchical clustering", )
    args = parser.parse_args()
    # print(args)  # keep for debugging args
    args.function(args)


def handle_configuration(args):
    # config_path = args.configuration
    if os.path.isfile(args.configuration):
        parse_configuration(args, args.configuration)
    elif os.path.isdir(os.path.abspath(args.configuration)):
        for filename in os.listdir(args.configuration):
            # print(filename)
            parse_configuration(args, os.path.join(args.configuration, filename))


def parse_configuration(args, filename):
    if filename.endswith(".ini"):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        for section in config.sections():
            args_copy = copy.copy(args)
            # print(section)
            args_copy.algorithm = config[section]["--algorithm"]  # sets algorithm from config file
            # print(args_copy.algorithm)
            args_copy.source = config[section]["--source"]
            args_copy.destination = config[section]["--destination"]
            if args_copy.algorithm == "hierarchical":
                args_copy.linkage = config[section]["--linkage"]
            print(args)
            call_algorithm(args_copy)
    else:
        print(args.configuration + " is not .ini type")


def call_algorithm(args):
    print(args)
    if args.algorithm == "hierarchical":
        # print(args.visualise)
        hierarchical.runClustering(args.source, args.destination, args.linkage, args.visualise)

    # call algorithm with these args
    elif args.algorithm == "imwkmeans":
        # TODO: call imwkmeans with other args
        None


if __name__ == "__main__":
    parse()
