#!/bin/bash

MODELS=(
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
)
BATCH_SIZES=(
    128
    64
    32
)

echo "Starting LFW experiments..."

PYTHON_SCRIPT="./scripts/lfw.py"
COMMON_ARGS="--fold_num -1 --device cuda:0"

NUM_EXPERIMENTS=${#MODELS[@]}
for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    MODEL=${MODELS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}

    COMMAND="python $PYTHON_SCRIPT --model $MODEL --batch_size $BATCH_SIZE $COMMON_ARGS"
    eval $COMMAND
done

echo "Starting AgeDB experiments..."

PYTHON_SCRIPT="./scripts/agedb.py"
COMMON_ARGS="--device cuda:0"

NUM_EXPERIMENTS=${#MODELS[@]}
for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    MODEL=${MODELS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}

    COMMAND="python $PYTHON_SCRIPT --model $MODEL --batch_size $BATCH_SIZE $COMMON_ARGS"
    eval $COMMAND
done

echo "Starting CASIA-Iris experiments..."

PYTHON_SCRIPT="./scripts/casia.py"
COMMON_ARGS="--num_pairs 20000 --random_seed 42 --device cuda:0"

NUM_EXPERIMENTS=${#MODELS[@]}
for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    MODEL=${MODELS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}

    COMMAND="python $PYTHON_SCRIPT --model $MODEL --batch_size $BATCH_SIZE $COMMON_ARGS"
    eval $COMMAND
done

echo "Starting FVC experiments..."

PYTHON_SCRIPT="./scripts/fvc.py"
COMMON_ARGS="--device cuda:0"

for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    MODEL=${MODELS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}

    COMMAND="python $PYTHON_SCRIPT --model $MODEL --batch_size $BATCH_SIZE $COMMON_ARGS"
    eval $COMMAND
done

echo "Starting CelebA experiments..."

PYTHON_SCRIPT="./scripts/celeba.py"
COMMON_ARGS="--device cuda:1 --partition_num 2"

NUM_EXPERIMENTS=${#MODELS[@]}
for ((i=0; i<NUM_EXPERIMENTS; i++)); do
    MODEL=${MODELS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}

    COMMAND="python $PYTHON_SCRIPT --model $MODEL --batch_size $BATCH_SIZE $COMMON_ARGS"
    eval $COMMAND
done