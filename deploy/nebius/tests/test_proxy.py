"""Opt-in real NGINX test: Docker + OpenSSL, ephemeral secrets, no cloud access."""

import concurrent.futures
import os
import secrets
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(os.getenv('RUN_PROXY_TESTS') != '1', reason='Set RUN_PROXY_TESTS=1; needs Docker')
ROOT = Path(__file__).parents[1]
NGINX = 'nginx:1.28-alpine@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236'


def docker(*args):
    result = subprocess.run(['docker', *map(str, args)], text=True, capture_output=True, check=True)
    return (result.stdout + (result.stderr if args[0] == 'logs' else '')).strip()


@pytest.fixture(scope='module')
def proxy(tmp_path_factory):
    directory = tmp_path_factory.mktemp('nebius-proxy')
    token, password = secrets.token_hex(32), secrets.token_urlsafe(24)
    (directory / 'endpoint_token').write_text(token)
    hashed = subprocess.check_output(['openssl', 'passwd', '-apr1', '-stdin'], input=password, text=True).strip()
    (directory / 'htpasswd').write_text('test-user:' + hashed + '\n')
    (directory / 'endpoint_token').chmod(0o600)
    (directory / 'htpasswd').chmod(0o600)

    # Root -> bridge -> issuer -> leaf: managed ingress uses this depth.
    def issue(name, issuer=None):
        key, csr, cert = (directory / (name + suffix) for suffix in ('.key', '.csr', '.crt'))
        subprocess.run(
            [
                'openssl',
                'req',
                '-new',
                '-newkey',
                'rsa:2048',
                '-nodes',
                '-subj',
                '/CN=' + name,
                '-keyout',
                str(key),
                '-out',
                str(csr),
            ],
            check=True,
            capture_output=True,
        )
        extensions = directory / (name + '.ext')
        extensions.write_text(
            'basicConstraints=critical,CA:FALSE\nsubjectAltName=DNS:upstream\n'
            'keyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n'
            if name == 'upstream'
            else 'basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\n'
        )
        args = [
            'openssl',
            'x509',
            '-req',
            '-in',
            str(csr),
            '-days',
            '1',
            '-set_serial',
            str(len(name) + (10 if issuer else 0)),
            '-extfile',
            str(extensions),
            '-out',
            str(cert),
        ]
        args += (
            ['-CA', str(directory / (issuer + '.crt')), '-CAkey', str(directory / (issuer + '.key'))]
            if issuer
            else ['-signkey', str(key)]
        )
        subprocess.run(args, check=True, capture_output=True)

    issue('root')
    issue('bridge', 'root')
    issue('issuer', 'bridge')
    issue('upstream', 'issuer')
    (directory / 'key.pem').write_bytes((directory / 'upstream.key').read_bytes())
    (directory / 'cert.pem').write_bytes(
        b''.join((directory / (name + '.crt')).read_bytes() for name in ('upstream', 'issuer', 'bridge'))
    )
    network = 'ls-nebius-test-' + uuid.uuid4().hex[:10]
    containers = []
    docker('network', 'create', network)
    try:
        containers.append(
            docker(
                'run',
                '-d',
                '--network',
                network,
                '--network-alias',
                'upstream',
                '--network-alias',
                'wrongname',
                '-v',
                f'{directory}:/fixtures:ro',
                '-v',
                f'{ROOT / "tests/fake_ingress.py"}:/server.py:ro',
                'python:3.11-alpine@sha256:0d55920083f1ce1e38ac292e2772f924b4f8bb4188d336c79bf66963039e6146',
                'python',
                '/server.py',
            )
        )

        def start(host='upstream', trust=True):
            args = [
                'run',
                '-d',
                '--network',
                network,
                '-p',
                '127.0.0.1::8080',
                '-e',
                f'NEBIUS_ENDPOINT_HOST={host}',
                '-e',
                'NEBIUS_ENDPOINT_PORT=8443',
                '-v',
                f'{directory / "endpoint_token"}:/run/secrets/endpoint_token:ro',
                '-v',
                f'{directory / "htpasswd"}:/run/secrets/htpasswd:ro',
                '-v',
                f'{ROOT / "proxy/start.sh"}:/etc/nebius/start.sh:ro',
                '-v',
                f'{ROOT / "proxy/nginx.conf.template"}:/etc/nebius/nginx.conf.template:ro',
            ]
            if trust:
                args.extend(['-v', f'{directory / "root.crt"}:/etc/ssl/certs/ca-certificates.crt:ro'])
            container = docker(*args, '--entrypoint', '/bin/sh', NGINX, '/etc/nebius/start.sh')
            containers.append(container)
            port = docker('port', container, '8080/tcp').rsplit(':', 1)[-1]
            url = f'http://127.0.0.1:{port}'
            for _ in range(100):
                try:
                    response = requests.get(url + '/health', timeout=1)
                    if response.status_code == 401:
                        if host != 'upstream' or not trust:
                            return url, container
                        # The proxy starts before the independent TLS fixture is ready.
                        if requests.get(url + '/health', auth=('test-user', password), timeout=1).status_code == 200:
                            return url, container
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            pytest.fail('Proxy did not become ready: ' + docker('logs', container))

        url, container = start()
        yield url, ('test-user', password), container, token, start
    finally:
        for container in containers:
            docker('rm', '-f', container)
        # Docker can briefly retain an endpoint after container removal.
        for attempt in range(20):
            try:
                docker('network', 'rm', network)
                break
            except subprocess.CalledProcessError:
                if attempt == 19:
                    raise
                time.sleep(0.2)


def test_real_proxy_translates_basic_through_full_tls_chain(proxy):
    url, auth, _, _, _ = proxy
    assert requests.get(url + '/health', timeout=5).status_code == 401
    assert requests.get(url + '/health', auth=('test-user', 'wrong'), timeout=5).status_code == 401
    assert requests.get(url + '/health', headers={'Authorization': 'Bearer wrong'}, timeout=5).status_code == 401
    assert requests.get(url + '/health', auth=auth, timeout=5).status_code == 200
    for route in ('setup', 'predict'):
        payload = {'project': '1.0', 'unicode': 'annotation ✓'}
        response = requests.post(
            url + '/' + route, json=payload, auth=auth, headers={'Cookie': 'must-not-forward'}, timeout=5
        )
        assert response.status_code == 200
        assert response.json() == {'path': '/' + route, 'payload': payload, 'cookie': None}


def test_routes_payload_limit_and_upstream_errors(proxy):
    url, auth, _, _, _ = proxy
    for route in ('webhook', 'train', 'delete', 'metrics', 'other'):
        assert requests.post(url + '/' + route, json={}, auth=auth, timeout=5).status_code == 404
    assert requests.get(url + '/predict', auth=auth, timeout=5).status_code == 403
    assert requests.post(url + '/health', json={}, auth=auth, timeout=5).status_code == 403
    body = b'x' * (16 * 1024 * 1024 + 1024)
    assert requests.post(url + '/predict', data=body, auth=auth, timeout=5).status_code == 413
    # Tasks with saved brush annotations exceed a few hundred KiB.
    assert requests.post(url + '/predict', json={'rle': list(range(100_000))}, auth=auth, timeout=5).status_code == 200
    assert requests.post(url + '/predict', json={'status': 503}, auth=auth, timeout=5).status_code == 503


def test_one_active_prediction_and_no_secret_in_logs(proxy):
    url, auth, container, token, _ = proxy

    def send():
        return requests.post(url + '/predict', json={'delay': 1}, auth=auth, timeout=5).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: send(), range(2)))
    assert sorted(results) == [200, 429]
    logs = docker('logs', container)
    assert token not in logs and auth[1] not in logs


def test_unauthenticated_request_does_not_take_prediction_slot(proxy):
    url, auth, _, _, _ = proxy
    host, port = url.removeprefix('http://').split(':')
    with socket.create_connection((host, int(port)), timeout=5) as held:
        # NGINX answers 401 but keeps the request open while it discards the unfinished body.
        held.sendall(b'POST /predict HTTP/1.1\r\nHost: proxy\r\nContent-Length: 10\r\n\r\nab')
        assert held.recv(64).startswith(b'HTTP/1.1 401')
        assert requests.post(url + '/predict', json={}, auth=auth, timeout=5).status_code == 200


@pytest.mark.parametrize('host,trust', [('wrongname', True), ('upstream', False)])
def test_upstream_tls_verification(proxy, host, trust):
    _, auth, _, _, start = proxy
    url, _ = start(host, trust)
    assert requests.get(url + '/health', auth=auth, timeout=5).status_code == 502
