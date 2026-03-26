#!/bin/bash

# Check if at least one ID is provided
if [ "$#" -eq 0 ]; then
    echo "Error: No input IDs provided."
    echo "Usage: $0 <id1> <id2> <id3> ..."
    echo "Example: $0 5 12 42"
    exit 1
fi
SPLIT="test"  # Default split
# Define the base directory for inputs
BASE_DIR="datasets/ScienceQA/data/scienceqa/images/${SPLIT}"

# Loop through all IDs provided as arguments
for ID in "$@"; do
    INPUT_PATH="${BASE_DIR}/${ID}"

    echo "=================================================="
    echo "Processing ID: ${ID}"
    echo "Input Path: ${INPUT_PATH}"
    echo "=================================================="

    # ----------------------------------------------------
    # Stage 1: BLIP-2
    # ----------------------------------------------------
    echo "[Stage 1/3] Running BLIP-2 (flan-t5)..."
    python src/CoE_Image_to_Text.py \
        --input "${INPUT_PATH}" \
        blip2 --blip2_version flan-t5
    
    # Check if Stage 1 succeeded, skip to next ID if failed
    if [ $? -ne 0 ]; then
        echo "Error: Stage 1 failed for ID ${ID}. Skipping..."
        continue
    fi

    # ----------------------------------------------------
    # Stage 2: Qwen2-VL
    # ----------------------------------------------------
    echo "[Stage 2/3] Running Qwen2-VL (2b)..."
    python src/CoE_Image_to_Text.py \
        --input "${INPUT_PATH}" \
        --previous_prefixes blip2-flan-t5 \
        qwen2-vl --qwen2vl_version 2b
    
    if [ $? -ne 0 ]; then
        echo "Error: Stage 2 failed for ID ${ID}. Skipping..."
        continue
    fi

    # ----------------------------------------------------
    # Stage 3: LLaVA
    # ----------------------------------------------------
    echo "[Stage 3/3] Running LLaVA (7b)..."
    python src/CoE_Image_to_Text.py \
        --input "${INPUT_PATH}" \
        --previous_prefixes blip2-flan-t5,qwen2vl2b \
        llava --llava_version 7b

    if [ $? -ne 0 ]; then
        echo "Error: Stage 3 failed for ID ${ID}. Skipping..."
        continue
    fi

    echo "Successfully completed all stages for ID: ${ID}"
    echo ""
done

echo "Batch processing finished!"