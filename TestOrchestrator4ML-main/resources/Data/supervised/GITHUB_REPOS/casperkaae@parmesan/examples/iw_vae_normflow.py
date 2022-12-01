# This code implementents a variational autoencoder using importance weighted
# sampling as described in Burda et al. 2015 "Importance Weighted Autoencoders"
# and the planar normalizing flow described in Rezende et al. 2015
# "Variational Inference with Normalizing Flows"
import theano
theano.config.floatX = 'float32'
import matplotlib
matplotlib.use('Agg')
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli
from parmesan.layers import SampleLayer, NormalizingPlanarFlowLayer, ListIndexLayer, NormalizeLayer, ScaleAndShiftLayer
from parmesan.datasets import load_mnist_realval, load_mnist_binarized
from parmesan.utils import log_mean_exp
import matplotlib.pyplot as plt
import shutil, gzip, os, cPickle, time, math, operator, argparse

filename_script = os.path.basename(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str,
        help="sampled or fixed binarized MNIST, sample|fixed", default="sample")
parser.add_argument("-eq_samples", type=int,
        help="number of samples for the expectation over q(z|x)", default=1)
parser.add_argument("-iw_samples", type=int,
        help="number of importance weighted samples", default=1)
parser.add_argument("-lr", type=float,
        help="learning rate", default=0.001)
parser.add_argument("-anneal_lr_factor", type=float,
        help="learning rate annealing factor", default=0.9995)
parser.add_argument("-anneal_lr_epoch", type=float,
        help="larning rate annealing start epoch", default=1000)
parser.add_argument("-batch_norm", type=str,
        help="batch normalization", default='true')
parser.add_argument("-outfolder", type=str,
        help="output folder", default=os.path.join("results", os.path.splitext(filename_script)[0]))
parser.add_argument("-nonlin_enc", type=str,
        help="encoder non-linearity", default="rectify")
parser.add_argument("-nonlin_dec", type=str,
        help="decoder non-linearity", default="rectify")
parser.add_argument("-nhidden", type=int,
        help="number of hidden units in deterministic layers", default=500)
parser.add_argument("-nlatent", type=int,
        help="number of stochastic latent units", default=100)
parser.add_argument("-nflows", type=int,
        help="length of normalizing flow", default=5)
parser.add_argument("-batch_size", type=int,
        help="batch size", default=100)
parser.add_argument("-nepochs", type=int,
        help="number of epochs to train", default=10000)
parser.add_argument("-eval_epoch", type=int,
        help="epochs between evaluation of test performance", default=10)


args = parser.parse_args()

def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'very_leaky_rectify':
        return lasagne.nonlinearities.very_leaky_rectify
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    else:
        raise ValueError('invalid non-linearity \'' + nonlin + '\'')

iw_samples = args.iw_samples   #number of importance weighted samples
eq_samples = args.eq_samples   #number of samples for the expectation over E_q(z|x)
lr = args.lr
anneal_lr_factor = args.anneal_lr_factor
anneal_lr_epoch = args.anneal_lr_epoch
batch_norm = args.batch_norm == 'true' or args.batch_norm == 'True'
res_out = args.outfolder
nonlin_enc = get_nonlin(args.nonlin_enc)
nonlin_dec = get_nonlin(args.nonlin_dec)
nhidden = args.nhidden
latent_size = args.nlatent
dataset = args.dataset
nflows = args.nflows
batch_size = args.batch_size
num_epochs = args.nepochs
eval_epoch = args.eval_epoch

assert dataset in ['sample','fixed'], "dataset must be sample|fixed"

np.random.seed(1234) # reproducibility

### SET UP LOGFILE AND OUTPUT FOLDER
if not os.path.exists(res_out):
    os.makedirs(res_out)

# write commandline parameters to header of logfile
args_dict = vars(args)
sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
description = []
description.append('######################################################')
description.append('# --Commandline Params--')
for name, val in sorted_args:
    description.append("# " + name + ":\t" + str(val))
description.append('######################################################')

shutil.copy(os.path.realpath(__file__), os.path.join(res_out, filename_script))
logfile = os.path.join(res_out, 'logfile.log')
model_out = os.path.join(res_out, 'model')
with open(logfile,'w') as f:
    for l in description:
        f.write(l + '\n')


sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_x = T.matrix('x')


def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)


### LOAD DATA AND SET UP SHARED VARIABLES
if dataset is 'sample':
    print "Using real valued MNIST dataset to binomial sample dataset after every epoch "
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    del train_t, valid_t, test_t
    preprocesses_dataset = bernoullisample
else:
    print "Using fixed binarized MNIST data"
    train_x, valid_x, test_x = load_mnist_binarized()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function

train_x = np.concatenate([train_x,valid_x])

train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)

num_features=train_x.shape[-1]

sh_x_train = theano.shared(preprocesses_dataset(train_x), borrow=True)
sh_x_test = theano.shared(preprocesses_dataset(test_x), borrow=True)


def batchnormlayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=None)
    l = NormalizeLayer(l,name="BN-" + name)
    l = ScaleAndShiftLayer(l,name="SaS-" + name)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity,name="Nonlin-" + name)
    return l

def normaldenselayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=nonlinearity)
    return l

if batch_norm:
    print "Using batch Normalization - The current implementation calculates " \
          "the BN constants on the complete dataset in one batch. This might " \
          "cause memory problems on some GFX's"
    denselayer = batchnormlayer
else:
    denselayer = normaldenselayer


### MODEL SETUP
# Recognition model q(z|x)
l_in = lasagne.layers.InputLayer((None, num_features))
l_enc_h1 = denselayer(l_in, num_units=nhidden, name='ENC_DENSE1', nonlinearity=nonlin_enc)
l_enc_h1 = denselayer(l_enc_h1, num_units=nhidden, name='ENC_DENSE2', nonlinearity=nonlin_enc)
l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_LOG_VAR')

#sample layer
l_z = SampleLayer(mean=l_mu, log_var=l_log_var, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

#Normalizing Flow
l_logdet_J = []
l_zk = l_z
for i in range(nflows):
    l_nf = NormalizingPlanarFlowLayer(l_zk)
    l_zk = ListIndexLayer(l_nf,index=0)
    l_logdet_J += [ListIndexLayer(l_nf,index=1)] #we need this for the cost function

# Generative model q(x|z)
l_dec_h1 = denselayer(l_zk, num_units=nhidden, name='DEC_DENSE2', nonlinearity=nonlin_dec)
l_dec_h1 = denselayer(l_dec_h1, num_units=nhidden, name='DEC_DENSE1', nonlinearity=nonlin_dec)
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=num_features, nonlinearity=lasagne.nonlinearities.sigmoid, name='X_MU')

# get output needed for evaluating of training i.e with noise if any
train_out = lasagne.layers.get_output(
    [l_z, l_zk, l_mu, l_log_var, l_dec_x_mu]+l_logdet_J, sym_x, deterministic=False
)
z_train = train_out[0]
zk_train = train_out[1]
z_mu_train = train_out[2]
z_log_var_train = train_out[3]
x_mu_train = train_out[4]
logdet_J_train = train_out[5:]

# get output needed for evaluating of testing i.e without noise
eval_out = lasagne.layers.get_output(
    [l_z, l_zk, l_mu, l_log_var, l_dec_x_mu]+l_logdet_J, sym_x, deterministic=True
)
z_eval = eval_out[0]
zk_eval = eval_out[1]
z_mu_eval = eval_out[2]
z_log_var_eval = eval_out[3]
x_mu_eval = eval_out[4]
logdet_J_eval = eval_out[5:]


def latent_gaussian_x_bernoulli(z0, zk, z0_mu, z0_log_var, logdet_J_list, x_mu, x, eq_samples, iw_samples, epsilon=1e-6):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z0: (batch_size*eq_samples*iw_samples, num_latent)
	zk: (batch_size*eq_samples*iw_samples, num_latent)
    z0_mu: (batch_size, num_latent)
    z0_log_var: (batch_size, num_latent)
    logdet_J_list: list of `nflows` elements, each with shape (batch_size*eq_samples*iw_samples)
    x_mu: (batch_size*eq_samples*iw_samples, num_features)
    x: (batch_size, num_features)

    Reference: Burda et al. 2015 "Importance Weighted Autoencoders"
    """

    # reshape the variables so batch_size, eq_samples and iw_samples are separate dimensions
    z0 = z0.reshape((-1, eq_samples, iw_samples, latent_size))
    zk = zk.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))

    for i in range(len(logdet_J_list)):
        logdet_J_list[i] = logdet_J_list[i].reshape((-1, eq_samples, iw_samples))

    # dimshuffle x, z_mu and z_log_var since we need to broadcast them when calculating the pdfs
    x = x.dimshuffle(0, 'x', 'x', 1)                    # size: (batch_size, eq_samples, iw_samples, num_features)
    z0_mu = z0_mu.dimshuffle(0, 'x', 'x', 1)            # size: (batch_size, eq_samples, iw_samples, num_latent)
    z0_log_var = z0_log_var.dimshuffle(0, 'x', 'x', 1)  # size: (batch_size, eq_samples, iw_samples, num_latent)

    # calculate LL components, note that the log_xyz() functions return log prob. for indepenedent components separately 
    # so we sum over feature/latent dimensions for multivariate pdfs
    log_q0z0_given_x = log_normal2(z0, z0_mu, z0_log_var).sum(axis=3)
    log_pzk = log_stdnormal(zk).sum(axis=3)
    log_px_given_zk = log_bernoulli(x, x_mu, epsilon).sum(axis=3)

    #normalizing flow loss
    sum_logdet_J = 0
    for logdet_J_k in logdet_J_list:
        sum_logdet_J += logdet_J_k

    # Calculate the LL using log-sum-exp to avoid underflow                                       all log_***                                       -> shape: (batch_size, eq_samples, iw_samples)
    LL = log_mean_exp(log_pzk + log_px_given_zk - log_q0z0_given_x + sum_logdet_J, axis=2)      # log-mean-exp over iw_samples dimension            -> shape: (batch_size, eq_samples)
    LL = T.mean(LL)                                                                             # average over eq_samples, batch_size dimensions    -> shape: ()

    return LL, T.mean(log_q0z0_given_x), T.mean(sum_logdet_J), T.mean(log_pzk), T.mean(log_px_given_zk)

# LOWER BOUNDS
LL_train, log_qz_given_x_train, sum_logdet_J_train, log_pz_train, log_px_given_z_train = latent_gaussian_x_bernoulli(
    z_train, zk_train, z_mu_train, z_log_var_train, logdet_J_train, x_mu_train, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

LL_eval, log_qz_given_x_eval, sum_logdet_J_eval, log_pz_eval, log_px_given_z_eval = latent_gaussian_x_bernoulli(
    z_eval, zk_eval, z_mu_eval, z_log_var_eval, logdet_J_eval, x_mu_eval, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

#some sanity checks that we can forward data through the model
X = np.ones((batch_size, 784), dtype=theano.config.floatX) # dummy data for testing the implementation

print "OUTPUT SIZE OF l_z using BS=%d, latent_size=%d, sym_iw_samples=%d, sym_eq_samples=%d --"\
      %(batch_size, latent_size, iw_samples, eq_samples), \
    lasagne.layers.get_output(l_z,sym_x).eval(
    {sym_x: X, sym_iw_samples: np.int32(iw_samples),
     sym_eq_samples: np.int32(eq_samples)}).shape

#print "log_pz_train", log_pz_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples),sym_eq_samples:np.int32(eq_samples)}).shape
#print "log_px_given_z_train", log_px_given_z_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape
#print "log_qz_given_x_train", log_qz_given_x_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape
#print "lower_bound_train", LL_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape

# get all parameters
params = lasagne.layers.get_all_params([l_dec_x_mu], trainable=True)
for p in params:
    print p, p.get_value().shape

# note the minus because we want to push up the lowerbound
grads = T.grad(-LL_train, params)
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

updates = lasagne.updates.adam(cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

# Helper symbolic variables to index into the shared train and test data
sym_index = T.iscalar('index')
sym_batch_size = T.iscalar('batch_size')
batch_slice = slice(sym_index * sym_batch_size, (sym_index + 1) * sym_batch_size)

train_model = theano.function([sym_index, sym_batch_size, sym_lr, sym_eq_samples, sym_iw_samples], [LL_train, log_qz_given_x_train, sum_logdet_J_train, log_pz_train, log_px_given_z_train, z_mu_train, z_log_var_train],
                              givens={sym_x: sh_x_train[batch_slice]},
                              updates=updates)

test_model = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [LL_eval, log_qz_given_x_eval, sum_logdet_J_eval, log_pz_eval, log_px_given_z_eval],
                              givens={sym_x: sh_x_test[batch_slice]})


if batch_norm:
    collect_out = lasagne.layers.get_output(l_dec_x_mu,sym_x, deterministic=True, collect=True)
    f_collect = theano.function([sym_eq_samples, sym_iw_samples],
                                [collect_out],
                                givens={sym_x: sh_x_train})

# Training and Testing functions
def train_epoch(lr, eq_samples, iw_samples, batch_size):
    n_train_batches = train_x.shape[0] / batch_size
    costs, log_qz_given_x, sum_logdet_J, log_pz, log_px_given_z, z_mu_train, z_log_var_train  = [],[],[],[],[],[],[]
    for i in range(n_train_batches):
        cost_batch, log_qz_given_x_batch, sum_logdet_J_batch, log_pz_batch, log_px_given_z_batch, z_mu_batch, z_log_var_batch = train_model(i, batch_size, lr, eq_samples, iw_samples)
        costs += [cost_batch]
        log_qz_given_x += [log_qz_given_x_batch]
        sum_logdet_J += [sum_logdet_J_batch]
        log_pz += [log_pz_batch]
        log_px_given_z += [log_px_given_z_batch]
        z_mu_train += [z_mu_batch]
        z_log_var_train += [z_log_var_batch]
    return np.mean(costs), np.mean(log_qz_given_x), np.mean(sum_logdet_J), np.mean(log_pz), np.mean(log_px_given_z), np.concatenate(z_mu_train), np.concatenate(z_log_var_train)

def test_epoch(eq_samples, iw_samples, batch_size):
    if batch_norm:
        _ = f_collect(1,1) #collect BN stats on train
    n_test_batches = test_x.shape[0] / batch_size
    costs, log_qz_given_x, sum_logdet_J, log_pz, log_px_given_z = [],[],[],[],[]
    for i in range(n_test_batches):
        cost_batch, log_qz_given_x_batch, sum_logdet_J_batch, log_pz_batch, log_px_given_z_batch = test_model(i, batch_size, eq_samples, iw_samples)
        costs += [cost_batch]
        log_qz_given_x += [log_qz_given_x_batch]
        sum_logdet_J += [sum_logdet_J_batch]
        log_pz += [log_pz_batch]
        log_px_given_z += [log_px_given_z_batch]
    return np.mean(costs), np.mean(log_qz_given_x), np.mean(sum_logdet_J), np.mean(log_pz), np.mean(log_px_given_z)

print "Training"

# TRAIN LOOP
# We have made some the code very verbose to make it easier to understand.
total_time_start = time.time()
costs_train, log_qz_given_x_train, sum_logdet_J_train, log_pz_train, log_px_given_z_train = [],[],[],[],[]
LL_test1, log_qz_given_x_test1, sum_logdet_J_test1, log_pz_test1, log_px_given_z_test1 = [],[],[],[],[]
LL_test5000, log_qz_given_x_test5000, sum_logdet_J_test5000, log_pz_test5000, log_px_given_z_test5000 = [],[],[],[],[]
xepochs = []
logvar_z_mu_train, logvar_z_var_train, meanvar_z_var_train = None,None,None
for epoch in range(1, 1+num_epochs):
    start = time.time()

    #shuffle train data and train model
    np.random.shuffle(train_x)
    sh_x_train.set_value(preprocesses_dataset(train_x))
    train_out = train_epoch(lr, eq_samples, iw_samples, batch_size)

    if np.isnan(train_out[0]):
        ValueError("NAN in train LL!")

    if epoch >= anneal_lr_epoch:
        #annealing learning rate
        lr = lr*anneal_lr_factor

    if epoch % eval_epoch == 0:
        t = time.time() - start

        costs_train += [train_out[0]]
        log_qz_given_x_train += [train_out[1]]
        sum_logdet_J_train += [train_out[2]]
        log_pz_train += [train_out[3]]
        log_px_given_z_train += [train_out[4]]
        z_mu_train = train_out[5]
        z_log_var_train = train_out[6]

        print "calculating LL eq=1, iw=5000"
        test_out5000 = test_epoch(1, 5000, batch_size=5) # smaller batch size to reduce memory requirements
        LL_test5000 += [test_out5000[0]]
        log_qz_given_x_test5000 += [test_out5000[1]]
        sum_logdet_J_test5000 += [test_out5000[2]]
        log_pz_test5000 += [test_out5000[3]]
        log_px_given_z_test5000 += [test_out5000[4]]

        print "calculating LL eq=1, iw=1"
        test_out1 = test_epoch(1, 1, batch_size=50)
        LL_test1 += [test_out1[0]]
        log_qz_given_x_test1 += [test_out1[1]]
        sum_logdet_J_test1 += [test_out1[2]]
        log_pz_test1 += [test_out1[3]]
        log_px_given_z_test1 += [test_out1[4]]

        xepochs += [epoch]

        line = "*Epoch=%d\tTime=%.2f\tLR=%.5f\teq_samples=%d\tiw_samples=%d\tnflows=%d\n" %(epoch, t, lr, eq_samples, iw_samples, nflows) + \
               "  TRAIN:\tCost=%.5f\tlogqK(zK|x)=%.5f\t= [logq0(z0|x)=%.5f - sum logdet J=%.5f]\tlogp(zK)=%.5f\tlogp(x|zK)=%.5f\n" %(costs_train[-1], log_qz_given_x_train[-1] - sum_logdet_J_train[-1], log_qz_given_x_train[-1], sum_logdet_J_train[-1], log_pz_train[-1], log_px_given_z_train[-1]) + \
               "  EVAL-L1:\tCost=%.5f\tlogqK(zK|x)=%.5f\t= [logq0(z0|x)=%.5f - sum logdet J=%.5f]\tlogp(zK)=%.5f\tlogp(x|zK)=%.5f\n" %(LL_test1[-1], log_qz_given_x_test1[-1] - sum_logdet_J_test1[-1], log_qz_given_x_test1[-1], sum_logdet_J_test1[-1], log_pz_test1[-1], log_px_given_z_test1[-1]) + \
               "  EVAL-L5000:\tCost=%.5f\tlogqK(zK|x)=%.5f\t= [logq0(z0|x)=%.5f - sum logdet J=%.5f]\tlogp(zK)=%.5f\tlogp(x|zK)=%.5f" %(LL_test5000[-1], log_qz_given_x_test5000[-1] - sum_logdet_J_test5000[-1], log_qz_given_x_test5000[-1], sum_logdet_J_test5000[-1], log_pz_test5000[-1], log_px_given_z_test5000[-1])
        print line
        with open(logfile,'a') as f:
            f.write(line + "\n")

        #save model every 100'th epochs
        if epoch % 100 == 0:
            all_params=lasagne.layers.get_all_params([l_dec_x_mu])
            f = gzip.open(model_out + 'epoch%i'%(epoch), 'wb')
            cPickle.dump(all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # BELOW THIS LINE IS A LOT OF BOOK KEEPING AND PLOTTING OF RESULTS
        _logvar_z_mu_train = np.log(np.var(z_mu_train,axis=0))
        _logvar_z_var_train = np.log(np.var(np.exp(z_log_var_train),axis=0))
        _meanvar_z_var_train = np.log(np.mean(np.exp(z_log_var_train),axis=0))

        if logvar_z_mu_train is None:
            logvar_z_mu_train = _logvar_z_mu_train[:,None]
            logvar_z_var_train = _logvar_z_var_train[:,None]
            meanvar_z_var_train = _meanvar_z_var_train[:,None]
        else:
            logvar_z_mu_train = np.concatenate([logvar_z_mu_train,_logvar_z_mu_train[:,None]],axis=1)
            logvar_z_var_train = np.concatenate([logvar_z_var_train, _logvar_z_var_train[:,None]],axis=1)
            meanvar_z_var_train = np.concatenate([meanvar_z_var_train, _meanvar_z_var_train[:,None]],axis=1)

        #plot results
        plt.figure(figsize=[12,12])
        plt.plot(xepochs,costs_train, label="LL")
        plt.plot(xepochs,log_qz_given_x_train, label="logq(z|x)")
        plt.plot(xepochs,log_pz_train, label="logp(z)")
        plt.plot(xepochs,log_px_given_z_train, label="logp(x|z)")
        plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
        plt.title('Train'), plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(res_out+'/train.png'),  plt.close()

        plt.figure(figsize=[12,12])
        plt.plot(xepochs,LL_test1, label="LL_k1")
        plt.plot(xepochs,log_qz_given_x_test1, label="logq(z|x)")
        plt.plot(xepochs,log_pz_test1, label="logp(z)")
        plt.plot(xepochs,log_px_given_z_test1, label="logp(x|z)")
        plt.title('Eval L1'), plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(res_out+'/eval_L1.png'),  plt.close()

        plt.figure(figsize=[12,12])
        plt.plot(xepochs,LL_test5000, label="LL_k5000")
        plt.plot(xepochs,log_qz_given_x_test5000, label="logq(z|x)")
        plt.plot(xepochs,log_pz_test5000, label="logp(z)")
        plt.plot(xepochs,log_px_given_z_test5000, label="logp(x|z)")
        plt.title('Eval L5000'), plt.xlabel('Epochs'), plt.ylabel('log()'), plt.grid('on')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(res_out+'/eval_L5000.png'),  plt.close()

        fig, ax = plt.subplots()
        data = logvar_z_mu_train
        heatmap = ax.pcolor(data, cmap=plt.cm.Greys)
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_xticklabels(xepochs, minor=False)
        plt.xlabel('Epochs'), plt.ylabel('#Latent Unit'), plt.title('train log(var(mu))'), plt.colorbar(heatmap)
        plt.savefig(res_out+'/train_logvar_z_mu_train.png'),  plt.close()

        fig, ax = plt.subplots()
        data = logvar_z_var_train
        heatmap = ax.pcolor(data, cmap=plt.cm.Greys)
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_xticklabels(xepochs, minor=False)
        plt.xlabel('Epochs'), plt.ylabel('#Latent Unit'), plt.title('train log(var(var))'), plt.colorbar(heatmap)
        plt.savefig(res_out+'/train_logvar_z_var_train.png'),  plt.close()

        fig, ax = plt.subplots()
        data = meanvar_z_var_train
        heatmap = ax.pcolor(data, cmap=plt.cm.Greys)
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_xticklabels(xepochs, minor=False)
        plt.xlabel('Epochs'), plt.ylabel('#Latent Unit'), plt.title('train log(mean(var))'), plt.colorbar(heatmap)
        plt.savefig(res_out+'/train_meanvar_z_var_train.png'),  plt.close()
