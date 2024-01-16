#!/bin/bash
set -e

pushd test
HAE_PATH=../build/

function error_hook() {
  echo "Tests failed."
  exit 1
}

trap error_hook ERR

$HAE_PATH/hae "a detailed description of food" < data/meta.txt -n 3 -hl > res_t1.txt
diff t1.txt res_t1.txt

cat data/meta.json | jq -c .[] | $HAE_PATH/hae "the time" -n 1 > res_t2.txt
diff t2.txt res_t2.txt
