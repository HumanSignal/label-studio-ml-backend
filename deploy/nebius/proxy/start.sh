#!/bin/sh
set -eu
# Validate values before inserting them into NGINX configuration syntax.
: "${NEBIUS_ENDPOINT_HOST:?Set the hostname from the managed HTTPS URL}"
NEBIUS_ENDPOINT_PORT=${NEBIUS_ENDPOINT_PORT:-443}
case "$NEBIUS_ENDPOINT_HOST" in
    ''|*[!a-zA-Z0-9.-]*|[-.]*|*[-.]) echo 'Invalid endpoint hostname' >&2; exit 1;;
esac
case "$NEBIUS_ENDPOINT_PORT" in
    ''|*[!0-9]*) echo 'Invalid endpoint port' >&2; exit 1;;
esac
[ "$NEBIUS_ENDPOINT_PORT" -ge 1 ] && [ "$NEBIUS_ENDPOINT_PORT" -le 65535 ]
ENDPOINT_BEARER_TOKEN=$(cat /run/secrets/endpoint_token)
case "$ENDPOINT_BEARER_TOKEN" in
    ''|*[!a-fA-F0-9]*) echo 'Use an openssl rand -hex 32 endpoint token' >&2; exit 1;;
esac
[ "${#ENDPOINT_BEARER_TOKEN}" -eq 64 ]
test -s /run/secrets/htpasswd
umask 077
# NGINX workers must read the password database without exposing the source secret.
cp /run/secrets/htpasswd /etc/nginx/htpasswd
chown nginx:nginx /etc/nginx/htpasswd
chmod 0400 /etc/nginx/htpasswd
export NEBIUS_ENDPOINT_HOST NEBIUS_ENDPOINT_PORT ENDPOINT_BEARER_TOKEN
envsubst '${NEBIUS_ENDPOINT_HOST} ${NEBIUS_ENDPOINT_PORT} ${ENDPOINT_BEARER_TOKEN}' \
    < /etc/nebius/nginx.conf.template > /etc/nginx/conf.d/default.conf
unset ENDPOINT_BEARER_TOKEN
exec nginx -g 'daemon off;'
