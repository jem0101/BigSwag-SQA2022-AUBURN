'''
Created on Nov 5, 2015

A module to mock caffe structures for testing

@author: kashefy
'''
from caffe.proto.caffe_pb2 import Datum
from .proto.caffe_pb2 import TRAIN, TEST
from . import io

class Net:
    def forward(self):
        return "mock this"

# class io:
#    @staticmethod
#    def array_to_datum(s):
#        return Datum()

