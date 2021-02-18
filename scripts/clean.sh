#/bin/bash

# Performs cleanup of the locally downloaded files (except cache).

script_dir=$(dirname $0)

cd $script_dir/..
rm -rf detectron model_data neuralcoref mmf vqa-maskrcnn-benchmark requirements.txt
