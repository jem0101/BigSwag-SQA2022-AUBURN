import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

import os
from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
term_fn="TERMINAL/"+started_datestring+".txt"


# REDIRECT PRINT ON TERMINAL TO FILE FOR DOCUMENTATION
# import sys
# orig_stdout = sys.stdout
# f = open(term_fn, 'w')
# sys.stdout = f

print("\n--------------------\nPyTorch training run\n\n"+str(started_datestring)+"\n--------------------\n\n")


parser = argparse.ArgumentParser(description='PyTorch PF RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/2017',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default= 'model.pt',
                    help='default name to save the final model')
parser.add_argument('--model_dir', type=str,  default= "models/"+str(started_datestring),
                    help='path to save ONGOING model')
args = parser.parse_args()

print("CORPUS: "+args.data)

#MODEL SAVE DIRECTORY
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
    print("INITIALIZING Directory: "+args.model_dir)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

# CHANGED: set train and test batch size to be same
eval_batch_size = 10#args.batch_size#10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
print("\nTraining batch size: "+str(args.batch_size)+"\nTest and validation batch size: "+str(eval_batch_size))

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("\n# of tokens in dictionary: "+str(ntokens))
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        clipped_lr = lr * clip_gradient(model, args.clip)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            


# Loop over epochs.
lr = args.lr
prev_val_loss = None
val_loss=0.0

if prev_val_loss and val_loss > prev_val_loss:
    if lr>0.01:
        lr /= 4

print ("Learning rate: "+str(lr))

#val_loss=0.0

for epoch in range(1, args.epochs+1):

    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    

    t1='-' * 89
    t2='\n| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | valid ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
    t3='-' * 89
    ta=t1+t2+t3
    print(ta)

    with open(term_fn, 'a+') as f:
        f.write(ta)

    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        if lr>0.001:
            lr /= 4
     
    prev_val_loss = val_loss
    
        # SAVE MODEL
    model_name= args.model_dir+'/model-{:s}-emsize-{:d}-nhid_{:d}-nlayers_{:d}-batch_size_{:d}-epoch_{:d}-loss_{:.2f}-ppl_{:.2f}'.format(args.model, args.emsize, args.nhid, args.nlayers, args.batch_size, epoch, val_loss, math.exp(val_loss))+'.pt'
    print ("SAVING: "+ model_name)
    print('=' * 89)
    print(" ")
    torch.save(model, model_name)
    
	


# Run on test data and save the model.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
if args.save != '':
    with open(args.save, 'wb') as f:
        torch.save(model, f)


# sys.stdout = orig_stdout
# f.close()