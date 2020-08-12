import argparse
import configparser
import os


# print(os.getcwd())
algorithm_list = ["alg1", "alg2", "alg3"]


def parse():
    parser = argparse.ArgumentParser()
    # use group = parser.add_mutually_exclusive_group() if needed
    parser.add_argument("-c",
                        "--configuration",
                        default=None,
                        help="Select configuration file/s", )
    parser.add_argument("-a",
                        "--algorithm",
                        choices=algorithm_list,
                        help="Select a clustering algorithm",)
    parser.add_argument("-s",
                        "--source",
                        default="input/location",  # change
                        help="Select the input source",)
    parser.add_argument("-d",
                        "--destination",
                        default="output/destination",  # change
                        help="Select output destination",)
    parser.add_argument("-v",
                        "--visualise",
                        default=None,
                        help="Select whether to visualise cluster results",)
    args = parser.parse_args()
    print(args)  # keep for debugging args

    if args.configuration:
        # config_path = args.configuration
        if os.path.isfile(args.configuration):
            if args.configuration.endswith(".ini"):  # refactor in future to decrease repetition
                config = configparser.ConfigParser(allow_no_value=True)
                config.read(args.configuration)
                for section in config.sections():
                    print(section)
                    args.algorithm = config[section]["--algorithm"]  # sets source from config file
                    print(args.algorithm)
        elif os.path.isdir(os.path.abspath(args.configuration)):
            for filename in os.listdir(args.configuration):
                print(filename)
                if filename.endswith(".ini"):
                    config = configparser.ConfigParser(allow_no_value=True)
                    config.read(os.path.join(args.configuration, filename))
                    for section in config.sections():
                        print(section)
                        args.algorithm = config[section]["--algorithm"]
                        print(args.algorithm)
                # run clustering job
                else:
                    continue


if __name__ == "__main__":
    parse()
