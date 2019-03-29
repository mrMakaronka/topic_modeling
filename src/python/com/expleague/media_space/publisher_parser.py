from typing import Optional
from urllib.parse import urlparse


class PublisherParser:
    @staticmethod
    def parse(url: str) -> Optional[str]:
        parse = urlparse(url)
        if parse.hostname is None:
            return None

        publisher = parse.hostname.replace('www.', '')
        if publisher == 'feedproxy.google.com':
            return None
        elif publisher == 'vk.com':
            publisher += parse.path
        elif publisher == 't.me':
            split = parse.path.split('/')
            publisher += ('/' + split[1])
        elif publisher == 'twitter.com':
            split = parse.path.split('/')
            publisher += ('/' + split[1])
        return publisher
