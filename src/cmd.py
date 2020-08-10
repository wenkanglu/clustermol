import argparse
import configparser
import os


# print(os.getcwd())
config_path = os.path.join("configs", "testconfig.ini")
# print(config_path)
config = configparser.ConfigParser()
config.read(config_path)

algorithm_list = ["alg1", "alg2", "alg3"]


def parse():
    parser = argparse.ArgumentParser()
    # use group = parser.add_mutually_exclusive_group() if needed
    parser.add_argument("-c",
                        "--configuration",
                        type=bool,
                        choices=[True, False],
                        default=False,
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
                        type=bool,
                        choices=[True, False],
                        default=False,
                        help="Select whether to visualise cluster results",)
    args = parser.parse_args()
    print(args)  # keep for debugging args

    if args.configuration:
        for section in config.sections():
            print(section)
            args.source = config[section]["--source"]  # sets source from config file
            print(args.source)


if __name__ == "__main__":
    parse()
