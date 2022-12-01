""" classifier metadata """
import os
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("s11-classifier").version
except pkg_resources.DistributionNotFound:
    __version__ = os.environ['VERSION']
