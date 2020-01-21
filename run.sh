#!/bin/bash

# run.sh

# --
#

lr="0.01 0.005 0.001 0.0005 0.0001"
prep="node_embedding"
aggr="attention attention2"
edge="identity"
mp="attention gate"

count="0"
for l in $lr; do
for p in $prep; do
for a in $aggr; do
for e in $edge; do
for m in $mp; do
python3 ./train.py \
    --problem-path ../../data/freebase/ \
    --problem yago \
    --epochs 1000 \
    --batch-size 2048 \
    --lr-init $l \
    --lr-schedule constant\
    --dropout 0.5\
    --batchnorm\
    --prep-class $p \
    --edgeupt-class $e \
    --aggregator-class $a \
    --log-interval 1\
    --mpaggr-class
    > "experiment/freebase/fb_"$count".txt" 
let count++
done
done
done
done
done

