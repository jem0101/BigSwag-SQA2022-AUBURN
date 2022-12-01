# -*- coding: utf-8 -*-

"""Common interface for show media content services"""
#
# Author: Andrea Stagi (aka 4ndreaSt4gi)
# 2010-08-01

import re
import logging
import requests

from libturpial.lib.interfaces.service import GenericService


class PreviewMediaService(GenericService):
    def __init__(self):
        self.log = logging.getLogger('Service')
        self.url_pattern = ""

    def do_service(self, url):
        raise NotImplementedError

    def _get_content_from_url(self, url):
        response = requests.get(url)
        return response.content

    def _get_id_from_url(self, url):
        parts = url.split("/")
        return parts[-1]

    def can_manage_url(self, url):
        regex = re.compile(self.url_pattern)
        if regex.search(url):
            return True
        return False
