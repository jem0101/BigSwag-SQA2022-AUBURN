#!/usr/bin/env python
# encode: utf-8

# Active Learning (Uncertainly Sampling)
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import sys, numpy
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

import optparse
parser = optparse.OptionParser()
#parser.add_option("-c", dest="corpus", help="corpus module name under nltk.corpus (e.g. brown, reuters)", default='brown')
#parser.add_option("-r", dest="testrate", type="float", help="rate of test dataset in corpus", default=0.1)
parser.add_option("--seed", dest="seed", type="int", help="random seed")
(opt, args) = parser.parse_args()
numpy.random.seed(opt.seed)

output = False

def activelearn(data, label, strategy):
    #print strategy

    N, D = data.shape
    train = list(range(D))
    pool = range(D,N)
    predict = None

    for i in xrange(30-D):
        if predict != None:
            if strategy == "random":
                x = numpy.random.randint(len(pool))
            elif strategy == "least confident":
                x = predict.max(axis=1).argmin()
            elif strategy == "margin sampling":
                predict.sort(axis=1)
                x = (numpy.exp(predict[:,-1])-numpy.exp(predict[:,-2])).argmin()
            elif strategy == "entropy-based":
                x = numpy.nan_to_num(numpy.exp(predict)*predict).sum(axis=1).argmin()
            train.append(pool[x])
            del pool[x]

        cl = LogisticRegression()
        #cl = LogisticRegression(C=0.1, penalty="l1")
        cl.fit(data[train,:], label[train])
        predict = cl.predict_log_proba(data[pool,:])
        log_likelihood = 0
        correct = 0
        for n, logprob in zip(pool,predict):
            c = label[n]
            log_likelihood += logprob[c]
            if c == logprob.argmax(): correct += 1

        Z = len(pool)
        precision = float(correct) / Z
        perplexity = numpy.exp(-log_likelihood / Z)
        if output:
            print "%d : %d / %d = %f, %f" % (len(train), correct, Z, precision, perplexity)

    #print data[train,:], label[train]

    if D==2:
        import matplotlib.pyplot as plt
        plt.plot(data[pool,0], data[pool,1], 'x', color="red")
        plt.plot(data[train,0], data[train,1], 'o', color="red")
        plt.title(strategy)
        plt.show()

    return precision, perplexity


D=10
N=1000
presicions = []
perplexities = []
for i in xrange(100):
    data = numpy.random.randn(N,D)
    label = numpy.zeros(N, dtype=int)
    for n in xrange(N):
        c = n % D
        data[n, c] += 2
        label[n] = c

    result = []
    result.append(activelearn(data, label, "random"))
    result.append(activelearn(data, label, "least confident"))
    result.append(activelearn(data, label, "margin sampling"))
    result.append(activelearn(data, label, "entropy-based"))

    x = numpy.array(result)
    presicions.append(x[:,0])
    perplexities.append(x[:,1])

print numpy.mean(presicions, axis=0)

