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
cmp t1.txt res_t1.txt

cat data/meta.json | jq -c .[] | $HAE_PATH/hae "the time" -n 1 > res_t2.txt
cmp t2.txt res_t2.txt

cat data/newline_at_start.txt | $HAE_PATH/hae "foo" > res_t3.txt
cmp t3.txt res_t3.txt

cat data/single_line.txt | $HAE_PATH/hae "foo" > res_t4.txt
cmp t4.txt res_t4.txt

cat data/empty_start.txt | $HAE_PATH/hae "foo" > res_t5.txt
cmp t5.txt res_t5.txt

cat data/short_paragraphs.txt | $HAE_PATH/hae "a door" > res_t6.txt
cmp t6.txt res_t6.txt