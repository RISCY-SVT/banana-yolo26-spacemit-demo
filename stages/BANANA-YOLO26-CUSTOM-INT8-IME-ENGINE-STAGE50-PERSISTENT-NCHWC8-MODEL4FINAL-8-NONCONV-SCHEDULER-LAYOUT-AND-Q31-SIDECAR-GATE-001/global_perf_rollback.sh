#!/bin/sh
set -eu
sudo -n sysctl -w kernel.perf_event_paranoid=2
sudo -n sysctl -w kernel.kptr_restrict=1
