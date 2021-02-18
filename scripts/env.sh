#!/bin/bash

# Paths needed to access mmf and maskrcnn modules.

script_dir=$(dirname $0)

PATH=$PATH:$script_dir/../vqa-maskrcnn-benchmark
PATH=$PATH:$script_dir/../mmf
