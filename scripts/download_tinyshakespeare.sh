#!/usr/bin/env bash
set -euo pipefail

export homedir=/home/ezrain/PycharmProjects/Transformer
mkdir -p ${homedir}/data/tinyshakespeare
cd ${homedir}/data/tinyshakespeare

curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

echo "Tiny Shakespeare downloaded to data/tinyshakespeare/input.txt"

