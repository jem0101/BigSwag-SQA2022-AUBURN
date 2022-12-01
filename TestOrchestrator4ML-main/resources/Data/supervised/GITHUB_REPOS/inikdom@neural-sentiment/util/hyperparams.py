'''
This is the main logic for serializing and deserializing dictionaries
of hyperparameters (for use in checkpoint restoration and sampling)
'''
import os
import pickle

class HyperParameterHandler(object):
    def __init__(self, path):
        self.file_path = os.path.join(path, "hyperparams.p")

    def save_params(self, dic):
        with open(self.file_path, 'wb') as handle:
            pickle.dump(dic, handle)

    def get_params(self):
        with open(self.file_path, 'rb') as handle:
            return pickle.load(handle)

    def exists(self):
        '''
        Checks if hyper parameter file exists
        '''
        return os.path.exists(self.file_path)

    def check_changed(self, new_params):
        if self.check_exists():
            old_params = self.get_params()
            return old_params["num_layers"] != new_params["num_layers"] or\
                old_params["hidden_size"] != new_params["hidden_size"] or\
                old_params["max_seq_length"] != new_params["max_seq_length"] or\
                old_params["max_vocab_size"] != new_params["max_vocab_size"]
        else:
            return False
