#!/bin/bash

set -e

if [ -z $DRY_RUN ]; then
    DRY_RUN=0
fi

langs=("ja" "ko" "zh_simplified" "zh_traditional")
models=("visual" "lookup")

for lang in ${langs[@]}
do
    for model in ${models[@]}
    do
        cmd="GPU=0 allennlp train \
            config/liu-acl17/wiki_title/${lang}/${model}.jsonnet \
            -s output/liu-acl17/wiki_title/${lang}/${model} --force"
        
        if [ $DRY_RUN -eq 1 ]; then
            echo $cmd
        else
            eval $cmd
        fi
    done
done
