"""TLS/Bearer stand-in for managed ingress; not a Nebius implementation."""

import json
import ssl
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOKEN = Path('/fixtures/endpoint_token').read_text().strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.respond({})

    def do_POST(self):
        self.respond(json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or '{}'))

    def respond(self, payload):
        if self.headers.get('Authorization') != 'Bearer ' + TOKEN:
            self.send_error(401)
            return
        time.sleep(payload.get('delay', 0))
        content = json.dumps({'path': self.path, 'payload': payload, 'cookie': self.headers.get('Cookie')}).encode()
        self.send_response(payload.get('status', 200))
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)


server = ThreadingHTTPServer(('0.0.0.0', 8443), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/fixtures/cert.pem', '/fixtures/key.pem')
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
