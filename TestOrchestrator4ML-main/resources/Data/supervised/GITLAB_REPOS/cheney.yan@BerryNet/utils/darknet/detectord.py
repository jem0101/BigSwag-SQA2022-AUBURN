# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2016 dt42.io. All Rights Reserved.
#
# 09-01-2016 joseph@dt42.io Initial version
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Simple image classification server with Inception.

The server monitors image_dir and run inferences on new images added to the
directory. Every image file should come with another empty file with '.done'
suffix to signal readiness. Inference result of a image can be read from the
'.txt' file of that image after '.txt.done' is spotted.

This is an example the server expects clients to do. Note the order.

# cp cat.jpg /run/image_dir
# touch /run/image_dir/cat.jpg.done

Clients should wait for appearance of 'cat.jpg.txt.done' before getting
result from 'cat.jpg.txt'.
"""


from __future__ import print_function
import os
import sys
import time
import signal
import argparse
import subprocess
import errno
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


def logging(*args):
    print("[%08.3f]" % time.time(), ' '.join(args))


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
        os.remove(_msg)
        logging(_msg, event.event_type)
        darknet.stdin.write(_msg.rstrip('.done').encode('utf8') + b'\n')

    # ignore all other types of events except 'modified'
    def on_created(self, event):
        self.process(event)


def check_pid(pid):
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM means no permission, and the process exists to deny the
            # access
            return True
        else:
            raise
    else:
        return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    pid = str(os.getpid())
    basename = os.path.splitext(os.path.basename(__file__))[0]
    ap.add_argument('input_dir')
    ap.add_argument(
        '-p', '--pid', default='/tmp/%s.pid' % basename,
        help='pid file path')
    ap.add_argument(
        '-fi', '--fifo', default='/tmp/acti_yolo',
        help='fifo pipe path')
    ap.add_argument(
        '-d', '--data', default='cfg/coco.data',
        help='fifo pipe path')
    ap.add_argument(
        '-c', '--config', default='cfg/yolo.cfg',
        help='fifo pipe path')
    ap.add_argument(
        '-w', '--weight', default='yolo.weights',
        help='fifo pipe path')
    args = vars(ap.parse_args())
    WATCH_DIR = os.path.abspath(args['input_dir'])
    FIFO_PIPE = os.path.abspath(args['fifo'])
    data = args['data']
    cfg = args['config']
    weight = args['weight']
    pidfile = args['pid']

    if os.path.isfile(pidfile):
        with open(pidfile) as f:
            prev_pid = int(f.readline())
            if check_pid(prev_pid):
                logging("{} already exists and process {} is still running, exists.".format(
                    pidfile, prev_pid))
                sys.exit(1)
            else:
                logging("{} exists but process {} died, clean it up.".format(pidfile, prev_pid))
                os.unlink(pidfile)

    with open(pidfile, 'w') as f:
        f.write(pid)

    logging("watch_dir: ", WATCH_DIR)
    logging("pid: ", pidfile)

    cmd = ['./darknet', 'detector', 'test', data, cfg, weight, '-out', '/usr/local/berrynet/dashboard/www/freeboard/snapshot']
    darknet = subprocess.Popen(cmd, bufsize=0,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    observer = Observer()
    observer.schedule(
        EventHandler(['*.jpg.done', '*.png.done']),
        path=WATCH_DIR, recursive=True)
    observer.start()
    try:
        darknet.wait()
    except KeyboardInterrupt:
        logging("Interrupt by user, clean up")
        os.kill(darknet.pid, signal.SIGKILL)
        os.unlink(pidfile)
