import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import mtbo

from hpolib.benchmarks.ml.svm_benchmark import SvmOnMnist, SvmOnVehicle, SvmOnCovertype, SvmOnAdult, SvmOnHiggs, SvmOnLetter
from hpolib.benchmarks.ml.residual_networks import ResidualNeuralNetworkOnCIFAR10
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnCIFAR10, ConvolutionalNeuralNetworkOnSVHN


run_id = int(sys.argv[1])
dataset = sys.argv[2]
seed = int(sys.argv[3])

rng = np.random.RandomState(seed)

if dataset == "mnist":
    f = SvmOnMnist(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "vehicle":
    f = SvmOnVehicle(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "covertype":
    f = SvmOnCovertype(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "adult":
    f = SvmOnAdult(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "letter":
    f = SvmOnLetter(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "higgs":
    f = SvmOnHiggs(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/svm_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "cifar10":
    f = ConvolutionalNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/cnn_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "svhn":
    f = ConvolutionalNeuralNetworkOnSVHN(rng=rng)
    num_iterations = 30
    output_path = "./experiments/fabolas/results/cnn_%s/mtbo_%d" % (dataset, run_id)
elif dataset == "res_net":
    f = ResidualNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 15
    output_path = "./experiments/fabolas/results/%s/mtbo_%d" % (dataset, run_id)

os.makedirs(output_path, exist_ok=True)


def objective(x, task):
    if task == 0:
        dataset_fraction = .25
    elif task == 1:
        dataset_fraction = 1

    res = f.objective_function(x, dataset_fraction=dataset_fraction)
    return res["function_value"], res["cost"]

info = f.get_meta_information()
bounds = np.array(info['bounds'])
results = mtbo(objective_function=objective,
               lower=bounds[:, 0], upper=bounds[:, 1],
               n_init=5, num_iterations=num_iterations, n_hypers=50,
               rng=rng, output_path=output_path)

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
current_inc = None
current_inc_val = None

key = "incumbents"

for inc in results["incumbents"]:
    print(inc)
    if current_inc == inc:
        test_error.append(current_inc_val)
    else:
        y = f.objective_function_test(inc)["function_value"]
        test_error.append(y)

        current_inc = inc
        current_inc_val = y
    print(current_inc_val)

    results["test_error"] = test_error

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
