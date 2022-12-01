import base64
import json

from datetime import datetime

import cv2
import numpy as np


def stringify_jpg(jpg_bytes):
    return base64.b64encode(jpg_bytes).decode('utf-8')


def destringify_jpg(stringified_jpg):
    """
    :return: JPEG bytes
    :rtype: bytes
    """
    return base64.b64decode(stringified_jpg.encode('utf-8'))


def jpg2bgr(jpg_bytes):
    """
    :return: BGR bytes
    :rtype: numpy array
    """
    array = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(array, flags=1)


def jpg2rgb(jpg_bytes):
    """
    :return: RGB bytes
    :rtype: numpy array
    """
    return cv2.cvtColor(jpg2bgr(jpg_bytes), cv2.COLOR_BGR2RGB)


def serialize_payload(json_object):
    return json.dumps(json_object)


def serialize_jpg(jpg_bytes):
    """Create Serialized JSON object consisting of image bytes and meta

    :param imarray: JPEG bytes
    :type imarray: bytes
    :return: serialized image JSON
    :rtype: string
    """
    obj = {}
    obj['timestamp'] = datetime.now().isoformat()
    obj['bytes'] = stringify_jpg(jpg_bytes)
    return json.dumps(obj)


def deserialize_payload(payload):
    return json.loads(payload)


#def deserialize_jpg(jpg_json):
#    """Deserialized JSON object created by josnify_image.
#
#    :param string :
#    :return:
#    :rtype:
#    """
#    return json.loads(jpg_json)


if __name__ == '__main__':
    im = cv2.imread('/home/debug/codes/darknet/data/dog.jpg')
    retval, jpg_bytes = cv2.imencode('.jpg', im)

    # size of stringified dog.jpg is 1.33x larger than original
    s_jpg = serialize_jpg(jpg_bytes)
    d_jpg = deserialize_payload(s_jpg)
    # TODO: Can we write JPEG bytes into file directly to prevent
    #       bytes -> numpy array -> decode RGB -> write encoded JPEG
    cv2.imwrite('/tmp/dog.jpg', jpg2bgr(destringify_jpg(d_jpg['bytes'])))
