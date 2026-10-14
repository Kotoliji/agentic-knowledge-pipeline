#!/usr/bin/env sh
# Regenerates the Python gRPC stubs. CI runs this and fails on any diff.
set -e
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I proto \
  --python_out=src --grpc_python_out=src --pyi_out=src \
  proto/akp/rpc/retrieval.proto
