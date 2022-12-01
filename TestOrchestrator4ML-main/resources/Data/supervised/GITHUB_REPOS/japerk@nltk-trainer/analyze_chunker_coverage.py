#!/usr/bin/env python
import argparse, collections, math
import nltk.corpus, nltk.corpus.reader, nltk.data, nltk.tag, nltk.metrics
from nltk.corpus.util import LazyCorpusLoader
from nltk_trainer import load_corpus_reader, load_model, simplify_wsj_tag
from nltk_trainer.chunking import chunkers
from nltk_trainer.chunking.transforms import node_label
from nltk_trainer.tagging import taggers

########################################
## command options & argument parsing ##
########################################

parser = argparse.ArgumentParser(description='Analyze a part-of-speech tagged corpus',
	formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('corpus',
	help='''The name of a tagged corpus included with NLTK, such as treebank,
brown, cess_esp, floresta, or the root path to a corpus directory,
which can be either an absolute path or relative to a nltk_data directory.''')
parser.add_argument('--tagger', default=None,
	help='''pickled tagger filename/path relative to an nltk_data directory
default is NLTK's default tagger''')
parser.add_argument('--chunker', default=nltk.chunk._MULTICLASS_NE_CHUNKER,
	help='''pickled chunker filename/path relative to an nltk_data directory
default is NLTK's default multiclass chunker''')
parser.add_argument('--trace', default=1, type=int,
	help='How much trace output you want, defaults to 1. 0 is no trace output.')
parser.add_argument('--score', action='store_true', default=False,
	help='Evaluate chunk score of chunker using corpus.chunked_sents()')

corpus_group = parser.add_argument_group('Corpus Reader Options')
corpus_group.add_argument('--reader', default=None,
	help='''Full module path to a corpus reader class, such as
nltk.corpus.reader.chunked.ChunkedCorpusReader''')
corpus_group.add_argument('--fileids', default=None,
	help='Specify fileids to load from corpus')
corpus_group.add_argument('--fraction', default=1.0, type=float,
	help='''The fraction of the corpus to use for testing coverage''')

if simplify_wsj_tag:
	corpus_group.add_argument('--simplify_tags', action='store_true', default=False,
		help='Use simplified tags')

args = parser.parse_args()

###################
## corpus reader ##
###################

corpus = load_corpus_reader(args.corpus, reader=args.reader, fileids=args.fileids)

if args.score and not hasattr(corpus, 'chunked_sents'):
	raise ValueError('%s does not support scoring' % args.corpus)

############
## tagger ##
############

if args.trace:
	print('loading tagger %s' % args.tagger)

if not args.tagger:
	tagger = nltk.tag._get_tagger()
elif args.tagger == 'pattern':
	tagger = taggers.PatternTagger()
else:
	tagger = load_model(args.tagger)

if args.trace:
	print('loading chunker %s' % args.chunker)

if args.chunker == 'pattern':
	chunker = chunkers.PatternChunker()
else:
	chunker = load_model(args.chunker)

#######################
## coverage analysis ##
#######################

if args.score:
	if args.trace:
		print('evaluating chunker score\n')
	
	chunked_sents = corpus.chunked_sents()
	
	if args.fraction != 1.0:
		cutoff = int(math.ceil(len(chunked_sents) * args.fraction))
		chunked_sents = chunked_sents[:cutoff]
	
	print(chunker.evaluate(chunked_sents))
	print('\n')

if args.trace:
	print('analyzing chunker coverage of %s with %s\n' % (args.corpus, chunker.__class__.__name__))

iobs_found = collections.defaultdict(int)
sents = corpus.sents()

if args.fraction != 1.0:
	cutoff = int(math.ceil(len(sents) * args.fraction))
	sents = sents[:cutoff]

for sent in sents:
	tree = chunker.parse(tagger.tag(sent))
	
	for child in tree.subtrees(lambda t: node_label(t) != 'S'):
		iobs_found[node_label(child)] += 1

iobs = iobs_found.keys()
justify = max(7, *[len(iob) for iob in iobs])

print('IOB'.center(justify) + '    Found  ')
print('='*justify + '  =========')

for iob in sorted(iobs):
	print('  '.join([iob.ljust(justify), str(iobs_found[iob]).rjust(9)]))

print('='*justify + '  =========')