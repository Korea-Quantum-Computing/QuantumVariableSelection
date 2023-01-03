
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Process simulation parameters.')
parser.add_argument('-n','--number_samples', default = 1000, type = int, help='Number of Samples')
parser.add_argument('-p','--number_features', default = 100, type = int, help='Number of Features')
parser.add_argument('-q','--number_influentials', default = 20, type = int, help='Number of Infulential Features')
parser.add_argument('-e','--epsilon', default = 30, type = int, help='Noise of Samples')
parser.add_argument('-cov','--covariance_parameter', default = 1, type = int, help='Magnitude of Covariance')
parser.add_argument('-lmbd','--lmbd', default = 0.1, type = int, help='Hyperparameter Controlling Convariance Mitigation')
parser.add_argument('-iter','--number_of_test', default = 100, type = int, help='a number of test')

if __name__ == '__main__':
    random_index = str(random.randrange(0,99999)).zfill(5)
    args = parser.parse_args()
    with open("result/simulation/test_argument"+random_index+".txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)