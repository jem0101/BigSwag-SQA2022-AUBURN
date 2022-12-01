from six.moves import urllib
import os
import zipfile
from collections import Counter
import tensorflow as tf
import numpy as np
from datetime import datetime

# Parameters for the model
VOCAB_SIZE = 50
EMBED_SIZE = 5 # Between 50 and 300 is good.
SKIP_WINDOW = 1

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = 'data/'
FILE_NAME = 'text8.zip'

def download(file_name, expected_bytes):
    file_path = DATA_FOLDER + file_name
    if (os.path.exists(file_path)):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if (file_stat.st_size == expected_bytes):
        print("Successfully downloaded the file", file_name)
    else:
        raise Exception("File " + file_name + " might be corrupted. You should try downloading it with a browser.")
    return file_path

def read_data(file_path):
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def build_vocab(words, vocab_size):
    dictionary = dict()
    count = [("UNK", -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    with open("processed/vocab_1000.tsv", "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if (index < 1000):
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    return [dictionary[word] if word in dictionary else 0 for word in words]
    
def build_cooccurrence_matrix(dictionary, index_words, context_window_size):
    n = len(dictionary)
    mat = np.zeros((n, n), dtype = np.float32)
    for center in index_words:
        for context in range(1, context_window_size+1):
            # process targets before the center word
            target = index_words[max(0, center - context)]
            mat[center, target] += 1
            # process targets after the center word
            target = index_words[min(center + context, len(index_words))]
            mat[center, target] += 1
    return mat

def process_data(vocab_size, skip_window):
    startTime = datetime.now()
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, index_dictionary = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    print("Processing data took: {0}".format(datetime.now() - startTime))
    startTime = datetime.now()
    mat = build_cooccurrence_matrix(dictionary, index_words, 2)
    print(np.shape(mat))
    print("Constructing matrix took: {0}".format(datetime.now() - startTime))
    return mat

mat = process_data(VOCAB_SIZE, SKIP_WINDOW)

def context_counting_model(mat, vocab_size, embed_size):
    cooccurrence_mat = tf.placeholder(tf.float32, shape = [vocab_size,
                                                          vocab_size],
                                     name = 'cooccurrence_matrix')
    s, u, v = tf.svd(cooccurrence_mat)
    smaller_mat = tf.matmul(u[:embed_size], tf.diag(s), b_is_sparse=True)
    another_mat = tf.matmul(smaller_mat, v[:, :embed_size])
    startTime = datetime.now()
    with tf.Session() as sess:
        print(mat)
        print(another_mat.eval(feed_dict = {cooccurrence_mat: mat}))
    print("Taking the svd took: {0}".format(datetime.now() - startTime))
    
context_counting_model(mat, VOCAB_SIZE, EMBED_SIZE)