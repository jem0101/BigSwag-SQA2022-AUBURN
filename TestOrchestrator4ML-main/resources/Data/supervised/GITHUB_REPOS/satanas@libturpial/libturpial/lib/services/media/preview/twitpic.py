# -*- coding: utf-8 -*-

"""Twitpic show media content service"""

from libturpial.api.models.media import Media
from libturpial.lib.services.media.preview.base import PreviewMediaService


class TwitpicMediaContent(PreviewMediaService):
    def __init__(self):
        PreviewMediaService.__init__(self)
        self.url_pattern = "(http(s)?://)?twitpic.com"

    def do_service(self, url):
        media_id = self._get_id_from_url(url)
        media_content_url = "http://twitpic.com/show/full/%s.png" % media_id
        info_ = {
            'source_url': media_content_url,
            'original_url': url,
        }
        rawimg = self._get_content_from_url(media_content_url)
        return Media.new_image(url, rawimg, info=info_)
