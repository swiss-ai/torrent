#!/bin/bash

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

pip install -U transformers

python -m sglang.launch_server "$@"
