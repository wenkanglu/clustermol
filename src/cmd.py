import argparse

algorithm_list = ["alg1", "alg2", "alg3"]


def parse():
    parser = argparse.ArgumentParser()
    # use group = parser.add_mutually_exclusive_group() if needed
    parser.add_argument("-a",
                        "--algorithm",
                        choices=algorithm_list,
                        help="Select a clustering algorithm",)
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    parse()
