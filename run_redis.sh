#!/usr/bin/env bash

set -ex

cd userdata && redis-server ../redis.main.conf
