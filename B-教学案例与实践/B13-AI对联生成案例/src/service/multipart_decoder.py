# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from requests_toolbelt import MultipartDecoder


def _split_on_find(content, bound):
    point = content.find(bound)
    return content[:point], content[point + len(bound):]


class NonMultipartContentTypeException(Exception):
    pass


class MultipartDecoderV2(MultipartDecoder):
    def _find_boundary(self):
        ct_info = tuple(x.strip() for x in self.content_type.split(b';'))
        mimetype = ct_info[0]
        if mimetype.split(b'/')[0] != b'multipart':
            raise NonMultipartContentTypeException(
                "Unexpected mimetype in content-type: '{0}'".format(mimetype)
            )
        for item in ct_info[1:]:
            attr, value = _split_on_find(
                item,
                b'='
            )
            if attr.lower() == b'boundary':
                self.boundary = value.strip(b'"')
