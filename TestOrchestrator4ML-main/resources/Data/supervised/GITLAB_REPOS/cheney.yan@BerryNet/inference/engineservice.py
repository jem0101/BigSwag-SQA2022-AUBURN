# Copyright 2017 DT42
#
# This file is part of BerryNet.
#
# BerryNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BerryNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BerryNet.  If not, see <http://www.gnu.org/licenses/>.

"""Engine service is a bridge between incoming data and inference engine.
"""


from __future__ import print_function

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import Queue

from dlmodelmgr import DLModelManager
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class DLEngine(object):
    def __init__(self):
        self.model_input_cache = []
        self.model_output_cache = []
        self.cache = {
            'model_input': [],
            'model_output': '',
            'model_output_filepath': ''
        }

    def create(self):
        # Workaround to posepone TensorFlow initialization.
        # If TF is initialized in __init__, and pass an engine instance
        # to engine service, TF session will stuck in run().
        pass

    def process_input(self, tensor):
        return tensor

    def inference(self, tensor):
        output = None
        return output

    def process_output(self, output):
        return output

    def cache_data(self, key, value):
        self.cache[key] = value

    def save_cache(self):
        with open(self.cache['model_output_filepath'], 'w') as f:
            f.write(str(self.cache['model_output']))


class EventHandler(PatternMatchingEventHandler):
    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        # the file will be processed there
        _msg = event.src_path
        self.image_queue.put(_msg.rstrip('.done'))
        os.remove(_msg)
        logging.debug(_msg + ' ' + event.event_type)

    # ignore all other types of events except 'modified'
    def on_created(self, event):
        self.process(event)


class EngineService(object):
    def __init__(self, service_name, engine):
        self.service_name = service_name
        self.engine = engine
        self.image_queue = Queue.Queue()
        self.event_handler = EventHandler(['*.jpg.done'])
        # NOTE: Increase object reference count (share memory)
        #       instead of object creation.
        self.event_handler.image_queue = self.image_queue

    def touch(self, fname, times=None):
        with open(fname, 'a'):
            os.utime(fname, times)

    def server(self):
        """Infinite loop serving inference requests"""

        logging.info(threading.current_thread().getName() + " is running")

        self.engine.create()
        while True:
            input_name = self.image_queue.get()
            image_data = cv2.imread(input_name).astype(np.float32)
            self.engine.cache_data('model_input', image_data)
            image_data = self.engine.process_input(image_data)

            output = self.engine.inference(image_data)
            model_outputs = self.engine.process_output(output)
            self.engine.cache_data('model_output', model_outputs)

            output_name = input_name + '.txt'
            output_done_name = output_name + '.done'
            self.engine.cache_data('model_output_filepath', output_name)
            self.engine.save_cache()
            self.touch(output_done_name)
            logging.debug(input_name + " classified!")

    def run(self, args):
        self.record_pid()

        # workaround the issue that SIGINT cannot be received (fork a child to
        # avoid blocking the main process in Thread.join()
        child_pid = os.fork()
        if child_pid == 0:  # child
            # observer handles event in a different thread
            observer = Observer()
            observer.schedule(self.event_handler, path=args['image_dir'])
            observer.start()
            self.server()
        else:  # parent
            try:
                os.wait()
            except KeyboardInterrupt:
                os.kill(child_pid, signal.SIGKILL)
                self.erase_pid()

    def record_pid(self):
        """Write a PID pidfile /tmp/<service_name>.pid.
        """
        pid = str(os.getpid())
        pidfile = '/tmp/{}.pid'.format(self.service_name)
        if os.path.isfile(pidfile):
            logging.critical("%s already exists, exiting" % pidfile)
            sys.exit(1)
        with open(pidfile, 'w') as f:
            f.write(pid)

    def erase_pid(self):
        pidfile = '/tmp/{}.pid'.format(self.service_name)
        os.unlink(pidfile)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    help='Model file path')
    ap.add_argument('--label',
                    help='Label file path')
    ap.add_argument('--model_package',
                    default='',
                    help='Model package name')
    ap.add_argument('--image_dir', required=True,
                    help='Path to image file')
    ap.add_argument('--service_name', required=True,
                    help='Engine service name used as PID filename')
    ap.add_argument('--num_top_predictions', default=5,
                    help='Display this many predictions')
    return vars(ap.parse_args())


if __name__ == '__main__':
    import movidius as mv

    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    if args['model_package'] != '':
        dlmm = DLModelManager()
        meta = dlmm.get_model_meta(args['model_package'])
        args['model'] = meta['model']
        args['label'] = meta['label']
    logging.debug('model filepath: ' + args['model'])
    logging.debug('label filepath: ' + args['label'])
    logging.debug('image_dir: ' + args['image_dir'])

    mvng = mv.MovidiusNeuralGraph(args['model'], args['label'])
    engine_service = EngineService(args['service_name'], mvng)
    engine_service.run(args)
