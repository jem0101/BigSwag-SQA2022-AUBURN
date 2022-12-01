import os
import os.path as pt
import sys
from collections import defaultdict

ROOT = pt.dirname(__file__)
parent = pt.abspath(pt.join(ROOT, os.pardir))

for p in (os.pardir, parent):
    try:
        sys.path.remove(p)
    except ValueError:
        pass
print(sys.path)

import numpy as np
import torch
import crumpets.torch.metrics
import random


def test_MSELossMetric():
    for i in range(10):

        errormetric = crumpets.torch.metrics.MSELossMetric()
        sum = 0
        N, C, H, W = (10, 3, 120, 100)

        for j in range(50):
            tensora = 10 * torch.rand(N, C, H, W).double().cuda()
            tensorb = 10 * torch.rand(N, C, H, W).double().cuda()
            sum += (tensora - tensorb).pow(2).sum()
            error2 = errormetric({"output": tensora, "target_image": tensorb})['mse']
            assert (sum / ((j + 1) * N * C * H * W) - error2).abs().item() <= 1e-13


def test_NSSMetric():
    for blur in np.arange(0.00005, 0.2, 0.008):
        N, C, H, W = (6, 1, 224, 224)
        errormetric = crumpets.torch.metrics.NSSMetric()
        sum = 0
        for j in range(30):
            groundtruth = [list(set([(random.randint(0, H - 1), (random.randint(0, W - 1))) for _ in range(300)])) for _
                           in range(N)]
            groundtensor = torch.zeros(N, C, H, W).double()
            targetsum = 0
            for (i, sample) in enumerate(groundtruth):
                for (pointx, pointy) in sample:
                    targetsum += 1
                    groundtensor[i, 0, pointx, pointy] = 1.0
            # blurred = cudaaugs.add_blur(groundtensor.cuda(), [{"blur" : blur}]*N).double()
            blurred = groundtensor + torch.normal(torch.zeros(N, C, H, W), 1.0).double()

            # print((groundtensor-blurred).pow(2).sum())
            metricNSS = errormetric({"output": blurred, "target_image": groundtensor})['nss']
            mean = blurred.view(N, -1).mean(dim=1).cpu()
            std = (blurred - mean.view(N, C, 1, 1)).view(N, -1).std(dim=1, unbiased=True).cpu()
            groundNSS = 0
            for (i, sample) in enumerate(groundtruth):
                for (pointx, pointy) in sample:
                    nss = (blurred[i, 0, pointx, pointy] - mean[i]).item()
                    nss /= std[i].item() + 1e-5
                    groundNSS += nss
            groundNSS /= (targetsum)
            groundNSS /= N
            sum += groundNSS
            print(abs((sum / ((j + 1)) - metricNSS.item())))
            assert abs((sum / ((j + 1)) - metricNSS.item())) <= 1e-13


# class NSSMetric
def test_NSSMetric2():
    for i in range(50):

        errormetric = crumpets.torch.metrics.NSSMetric()
        sum = 0
        N, C, H, W = (10, 3, 120, 100)

        for j in range(50):
            tensora = 10 * torch.rand(N, C, H, W).double().cuda()
            tensorb = 10 * torch.rand(N, C, H, W).double().cuda()
            output = tensora.view(N, C, -1)
            mean = output.mean(dim=2).view(N, C, 1, 1)
            std = (output.std(dim=2) + 1e-5).view(N, C, 1, 1)
            NSS = (tensora - mean)
            NSS = NSS / std

            sum += ((NSS * tensorb).sum() / (tensorb.sum() * N))
            error2 = errormetric({"output": tensora, "target_image": tensorb})['nss']
            # print(error2)
            # print(sum/(j+1))
            assert (sum / ((j + 1)) - error2).abs().item() <= 1e-13


# class averageMetric

def test_averageMetrics():
    for i in range(50):

        errormetric = crumpets.torch.metrics.AverageMetric(output_key="output")
        sum = 0
        N, C, H, W = (10, 3, 120, 100)

        for j in range(50):
            tensora = 0.3 + 10 * torch.rand(N, C, H, W).double().cuda()
            sum += tensora.sum()
            error2 = errormetric({"output": tensora.sum(), })['average_metric']
            assert (sum / ((j + 1)) - error2).abs().item() <= 1e-13


#
def test_accuracyMetric():
    print("this test fails with torch versions <1.1.0 because of the use of torch"+
            ".nn.functional.one_hot. Check if torch.nn.functional.one_hot was found")
    N = 10
    top_k = (1, 3, 5, 7, 9)
    Cs = (9, 20, 50, 100)
    for C in Cs:
        for i in range(10):

            metric = crumpets.torch.metrics.AccuracyMetric(top_k=top_k, output_key="output")
            sumcorrect = defaultdict(lambda: 0)
            count = 0

            for j in range(50):
                tensorpredict = torch.rand(N, C)
                tensorground = torch.tensor([random.randint(0, C - 1) for _ in range(N)]).view(N, 1)
                sorted = tensorpredict.sort(1, descending=True)[1]
                for k in top_k:
                    sumcorrect[k] += sorted[:, :k].eq(tensorground).sum()
                count += N

                acc = metric({"output": tensorpredict, "label": tensorground})
                for k in top_k:
                    assert float(sumcorrect[k]) / count == acc['top-%d acc' % k]


def test_confusionMatrix():
    N = 1000
    nclasses = 10
    for _ in range(100):
        metric = crumpets.torch.metrics.ConfusionMatrix(nclasses=nclasses, output_key="output", target_key="target")
        matrix = torch.zeros((nclasses, nclasses)).int()
        targetlist = list(range(nclasses)) * N
        outputlist = list(range(nclasses)) * N
        targetlist2 = list(range(nclasses)) * N
        outputlist2 = list(range(nclasses)) * N
        random.shuffle(outputlist2)
        random.shuffle(outputlist)
        for output, target, output2, target2 in zip(outputlist, targetlist, outputlist2, targetlist2):
            metricoutput = torch.nn.functional.one_hot(torch.tensor([output, output2]), num_classes=nclasses)
            metrictarget = torch.tensor([target, target2])
            # print(metricoutput)
            # print(metrictarget)
            metric({'output': metricoutput, 'target': metrictarget})
            matrix[target, output] += 1
            matrix[target2, output2] += 1
        assert (1 - (torch.from_numpy(metric.cmat).int() == matrix).int()).sum().item() == 0


class __Metric(crumpets.torch.metrics.Metric):
    def __init__(self, name):
        self.called = 0
        self.name = name

    def __call__(self, sample):
        self.called += 1
        return self.value()

    def value(self):
        return {"called " + self.name: self.called}


def test_CombinedMetric():
    NUM_METRICS = 100
    ITERATIONS = 200
    metrics = [__Metric(name=str(i)) for i in range(NUM_METRICS)]
    metric = crumpets.torch.metrics.CombinedMetric(metrics)
    for i in range(ITERATIONS):
        result = metric(None)
        for key, value in result.items():
            assert value == i + 1
