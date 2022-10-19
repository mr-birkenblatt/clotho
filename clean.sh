#!/usr/bin/env bash

set -ex

if command -v redis-cli &> /dev/null; then
    # FIXME also make make command
    redis-cli \
        "EVAL" \
        "for _,k in ipairs(redis.call('keys', KEYS[1])) do redis.call('del', k) end" \
        1 \
        'api:salt:*'
fi
