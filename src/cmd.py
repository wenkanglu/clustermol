import argparse

algorithm_list = ["alg1", "alg2", "alg3"]


def parse():
    parser = argparse.ArgumentParser()
    # use group = parser.add_mutually_exclusive_group() if needed
    parser.add_argument("-a",
                        "--algorithm",
                        choices=algorithm_list,
                        required=True,
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


if __name__ == "__main__":
    parse()
