#!/bin/bash

# Check if at least one ID is provided
if [ "$#" -eq 0 ]; then
    echo "Error: No input IDs provided."
    echo "Usage: $0 <id1> <id2> <id3> ..."
    echo "Example: $0 5 12 42"
    exit 1
fi

# ===== Configurable Parameters =====
REPO_ROOT="."
SCIENCEQA_ROOT="datasets/ScienceQA/data/scienceqa"
SPLIT="test"
OLLAMA_MODEL="deepseek-r1:7b"

# KG Parameters
KG_WORKING_DIR_PREFIX="tmp_lightkg"
KG_LLM_MODEL="deepseek-r1:7b"
KG_QUERY_MODE="hybrid"

# RAG Parameters
RAG_SOURCE="all"
RAG_CHUNK_BY="sentence"
RAG_TOPK=5

# Output Directory
OUTPUT_DIR="result_all"
mkdir -p "${OUTPUT_DIR}"

# ===== Main Loop =====
for ID in "$@"; do
    echo "=================================================="
    echo "Running Evaluation for ID: ${ID}"
    echo "=================================================="

    # ----------------------------------------------------
    # 1) KG Mode
    # ----------------------------------------------------
    echo "[1/3] Running KG Mode..."
    python src/evaluate_scienceqa_rag_vs_kg.py \
        --repo_root "${REPO_ROOT}" \
        --scienceqa_root "${SCIENCEQA_ROOT}" \
        --split "${SPLIT}" \
        --problem_ids "${ID}" \
        --mode kg \
        --auto_build \
        --kg_working_dir "${KG_WORKING_DIR_PREFIX}_${ID}" \
        --kg_llm_model "${KG_LLM_MODEL}" \
        --kg_query_mode "${KG_QUERY_MODE}" \
        --ollama_model "${OLLAMA_MODEL}" \
        --output "${OUTPUT_DIR}/eval_${SPLIT}${ID}_kg.json"

    # Check execution status
    if [ $? -ne 0 ]; then
        echo "Error: KG Mode failed for ID ${ID}."
    fi

    # ----------------------------------------------------
    # 2) RAG Mode
    # ----------------------------------------------------
    echo "[2/3] Running RAG Mode..."
    python src/evaluate_scienceqa_rag_vs_kg.py \
        --repo_root "${REPO_ROOT}" \
        --scienceqa_root "${SCIENCEQA_ROOT}" \
        --split "${SPLIT}" \
        --problem_ids "${ID}" \
        --mode rag \
        --auto_build \
        --rag_source "${RAG_SOURCE}" \
        --rag_chunk_by "${RAG_CHUNK_BY}" \
        --rag_topk "${RAG_TOPK}" \
        --ollama_model "${OLLAMA_MODEL}" \
        --output "${OUTPUT_DIR}/eval_${SPLIT}${ID}_rag.json"

    if [ $? -ne 0 ]; then
        echo "Error: RAG Mode failed for ID ${ID}."
    fi

    # ----------------------------------------------------
    # 3) Baseline Mode
    # ----------------------------------------------------
    echo "[3/3] Running Baseline Mode..."
    python src/evaluate_scienceqa_rag_vs_kg.py \
        --repo_root "${REPO_ROOT}" \
        --scienceqa_root "${SCIENCEQA_ROOT}" \
        --split "${SPLIT}" \
        --problem_ids "${ID}" \
        --mode baseline \
        --ollama_model "${OLLAMA_MODEL}" \
        --output "${OUTPUT_DIR}/eval_${SPLIT}${ID}_baseline.json"

    if [ $? -ne 0 ]; then
        echo "Error: Baseline Mode failed for ID ${ID}."
    fi

    echo "Finished processing ID: ${ID}"
    echo ""
done

echo "Batch evaluation finished!"