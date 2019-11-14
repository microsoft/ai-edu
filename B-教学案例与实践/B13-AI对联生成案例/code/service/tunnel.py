# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import requests
import ssl
from io import BytesIO

class TunnelHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        """Serve a GET request."""
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if parts.path.endswith('/'):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return

        f = self.send_head()
        if f:
            try:
                self.copyfile(f, self.wfile)
            except:
                print('Failed to Copy File...')
            finally:
                f.close()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        metadata = self.rfile.read(content_length)

        if metadata[:2] == b'--':
            global index, lock_id, server_num, port0, port1, ip0, ip1
            tmp_index = 0
            if lock_id.acquire():
                index = (index + 1) % server_num
                tmp_index = index
                lock_id.release()
            ip = eval('ip{}'.format(tmp_index))
            port = eval('port{}'.format(tmp_index))
            print('request server {}:{}'.format(ip, port))
            r = requests.post(url='http://{}:{}'.format(ip, port), data=metadata)
        else:
            r = requests.post(url='http://127.0.0.1:{}'.format(8003), data=metadata)

        self.send_response(200)
        self.end_headers()
        message = r.text
        response = BytesIO()
        response.write(message.encode())
        self.wfile.write(response.getvalue())

    def handle_one_request(self):
        """Handle a single HTTP request.
        You normally don't need to override this method; see the class
        __doc__ string for information on how to handle specific HTTP
        commands such as GET and POST.
        """
        try:
            self.raw_requestline = self.rfile.readline(65537)
            if len(self.raw_requestline) > 65536:
                self.requestline = ''
                self.request_version = ''
                self.command = ''
                self.send_error(414)
                return
            if not self.raw_requestline:
                self.close_connection = 1
                return
            if not self.parse_request():
                # An error code has been sent, just exit
                return
            mname = 'do_' + self.command
            if not hasattr(self, mname):
                self.send_error(501, "Unsupported method (%r)" % self.command)
                return
            method = getattr(self, mname)
            method()
            self.wfile.flush()  # actually send the response if not already done.
        except socket.timeout as e:
            # a read or a write timed out.  Discard this connection
            self.log_error("Request timed out: %r", e)
            self.close_connection = 1
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def run(server_class=ThreadedHTTPServer, handler_class=TunnelHandler, port=8000, certfile=None):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    if certfile:
        httpd.socket = ssl.wrap_socket(httpd.socket, certfile, server_side=True)
    else:
        httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True)
    print('Starting tunnel httpd...')

    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv
    lock_id = threading.Lock()
    index = 0
    ip0 = '127.0.0.1'
    ip1 = '127.0.0.1'
    port0 = 8001
    port1 = 8002
    server_num = 2
    if len(argv) == 3:
        run(port=int(argv[1]), certfile=argv[2])
    elif len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run(port=8000)
