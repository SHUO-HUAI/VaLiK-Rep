#!/bin/bash

# Check if at least one ID is provided
if [ "$#" -eq 0 ]; then
    echo "Error: No input IDs provided."
    echo "Usage: $0 <id1> <id2> <id3> ..."
    echo "Example: $0 5 12 42"
    exit 1
fi

SPLIT="test"  # Default split

# Define base parameters
BASE_DIR="datasets/ScienceQA/data/scienceqa/images/${SPLIT}"
THRESHOLD="0.20"
MODE="sentence"
TEXT_SUFFIX=".blip2-flan-t5.qwen2vl2b.llava7b.txt"

for ID in "$@"; do
    echo "=================================================="
    echo "Running Verification for ID: ${ID}"
    echo "=================================================="

    DIR_PATH="${BASE_DIR}/${ID}"
    
    # Check if the ID directory exists
    if [ ! -d "${DIR_PATH}" ]; then
        echo "Error: Directory ${DIR_PATH} does not exist. Skipping..."
        continue
    fi

    # Enable nullglob (ignore empty matches) and nocaseglob (case-insensitive)
    shopt -s nullglob nocaseglob
    
    # Find all matching image files in the directory
    IMAGES=("${DIR_PATH}/"*.jpg "${DIR_PATH}/"*.jpeg "${DIR_PATH}/"*.png)
    
    # Revert shopt settings to default
    shopt -u nullglob nocaseglob

    # Check if any images were found
    if [ ${#IMAGES[@]} -eq 0 ]; then
        echo "Warning: No image files (jpg/jpeg/png) found in ${DIR_PATH}. Skipping..."
        continue
    fi

    # Process each found image
    for IMAGE_PATH in "${IMAGES[@]}"; do
        # Extract filename without extension (e.g., 'figure1' from 'figure1.png')
        BASENAME=$(basename -- "$IMAGE_PATH")
        FILENAME_NO_EXT="${BASENAME%.*}"
        
        # Construct the expected text file path
        TEXT_PATH="${DIR_PATH}/${FILENAME_NO_EXT}${TEXT_SUFFIX}"

        # Ensure the corresponding text file exists
        if [ ! -f "${TEXT_PATH}" ]; then
            echo "Warning: Matching text file not found for ${BASENAME}. Skipping..."
            continue
        fi

        echo "Processing Image: ${BASENAME}"
        
        # Run the similarity verification script
        python src/Prune/similarity_verification.py \
            --image_path "${IMAGE_PATH}" \
            --text_path "${TEXT_PATH}" \
            --threshold "${THRESHOLD}" \
            --mode "${MODE}"

        # Check execution status
        if [ $? -ne 0 ]; then
            echo "Error: Verification failed for image ${BASENAME} in ID ${ID}."
            continue
        fi
        
        echo "Successfully verified: ${BASENAME}"
    done
    echo ""
done

echo "Batch verification finished!"