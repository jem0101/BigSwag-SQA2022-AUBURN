import os
import random

import bottle
from bottle import request, redirect, static_file, mako_view as view

import tweedr
from tweedr.lib.text import token_re
from tweedr.ml.crf.classifier import CRF
from tweedr.ml.features import featurize
from tweedr.ml.features.sets import crf_feature_functions
from tweedr.models import DBSession, TokenizedLabel

import logging
logger = logging.getLogger(__name__)

# tell bottle where to look for templates
# We use Mako templates (*.mako) that are in the templates/ directory in the package root.
# There are also Handlebars (*.bars) templates in there, but those are rendered on the client-side.
bottle.TEMPLATE_PATH.append(os.path.join(tweedr.root, 'templates'))

# this is the primary export
app = bottle.Bottle()

# globals are messy, but we don't to retrain a tagger for every request
logger.debug('initializing %s (training or loading CRF using defaults)', __name__)
GLOBALS = dict(tagger=CRF.default(crf_feature_functions))


@app.get('/')
def root():
    redirect('/crf')


@app.get('/crf')
@view('crf.mako')
def index():
    # effectively static; all the fun stuff happens in the template
    return dict()


@app.get('/tokenized_labels/sample')
def tokenized_labels_sample():
    total = DBSession.query(TokenizedLabel).count()
    index = random.randrange(total)
    logger.debug('/tokenized_labels/sample: choosing #%d out of %d', index, total)
    tokenized_label = DBSession.query(TokenizedLabel).offset(index).limit(1).first()
    return tokenized_label.__json__()


@app.post('/tagger/tag')
def tagger_tag():
    # For bottle >= 0.10, request.forms.xyz attributes return unicode strings
    # and an empty string if decoding fails.
    text = request.forms.text
    tokens = token_re.findall(text.encode('utf8'))

    tokens_features = map(list, featurize(tokens, crf_feature_functions))
    tagger = GLOBALS['tagger']
    labels = tagger.predict([tokens_features])[0]

    sequences = [
        {'name': 'tokens', 'values': tokens},
        {'name': 'labels', 'values': labels},
    ]
    for feature_function in crf_feature_functions:
        sequences.append({
            'name': feature_function.__name__,
            'values': [', '.join(features) for features in feature_function(tokens)]})

    return {'sequences': sequences}


@app.route('/tagger/retrain')
def tagger_retrain():
    GLOBALS['tagger'] = CRF.default(crf_feature_functions, retrain=True)
    return dict(success=True)


@app.route('/static/<filepath:path>')
def serve_static_file(filepath):
    return static_file(filepath, os.path.join(tweedr.root, 'static'))


@app.route('/templates/<filepath:path>')
def serve_templates_file(filepath):
    return static_file(filepath, os.path.join(tweedr.root, 'templates'))
