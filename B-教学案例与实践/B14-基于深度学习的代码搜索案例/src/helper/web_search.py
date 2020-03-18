# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import tornado.ioloop
import tornado.web
import tornado.log
import json


class WebSearchHandler(tornado.web.RequestHandler):
    def initialize(self, handler):
        # noinspection PyAttributeOutsideInit
        self.handler = handler
        
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        query = self.get_query_argument('q')
        count = 5
        try:
            count = int(self.get_query_argument('n', '5'))
            if count <= 0:
                count = 5
        except ValueError:
            pass
        results = self.handler(query=query, count=count)
        self.write(json.dumps({
            'query': query,
            'results': results
        }))


def run_web_search(host, port, handler):
    app = tornado.web.Application([
        (r'/search', WebSearchHandler, dict(handler=handler))
    ])
    print('start server at http://%s:%d' % (host, port))
    app.listen(port, host)
    tornado.log.enable_pretty_logging()
    tornado.ioloop.IOLoop.current().start()
