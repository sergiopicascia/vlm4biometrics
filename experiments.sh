#!/bin/bash

# ==========================================
# CONFIGURATION
# Format: "Model_HuggingFace_Path:Batch_Size"
# ==========================================

# You can comment out specific lines to skip them
MODELS_CONFIG=(
    # --- GEMMA ---
    "google/gemma-3-4b-it:96"
    "google/gemma-3-12b-it:64"
    "google/gemma-3-27b-it:32"
    
    # --- QWEN ---
    "Qwen/Qwen3-VL-2B-Instruct:96"
    "Qwen/Qwen3-VL-4B-Instruct:96"
    "Qwen/Qwen3-VL-8B-Instruct:64"
    "Qwen/Qwen3-VL-30B-A3B-Instruct:32"
    "Qwen/Qwen3-VL-32B-Instruct:16"
    
    # --- INTERNVL ---
    "OpenGVLab/InternVL3_5-1B-HF:128"
    "OpenGVLab/InternVL3_5-2B-HF:96"
    "OpenGVLab/InternVL3_5-4B-HF:64"
    "OpenGVLab/InternVL3_5-8B-HF:64"
    "OpenGVLab/InternVL3_5-14B-HF:32"
    "OpenGVLab/InternVL3_5-38B-HF:16"
    "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF:32"
    "OpenGVLab/InternVL3_5-30B-A3B-HF:32"
)

# Filter to run specific families only (leave empty to run all)
FILTER=""
DEVICE="cuda:0"

# Set these to "true" or "false" to control which experiment runs
RUN_LOGPROBS=true
RUN_MCQ=true

# ==========================================
# HELPER FUNCTION
# ==========================================

run_experiment_set() {
    local dataset_name=$1      # e.g., "LFW"
    local script_prefix=$2     # e.g., "lfw"
    local extra_args=$3        # e.g., "--num_pairs 20000"

    echo "========================================"
    echo "Processing Dataset: $dataset_name"
    echo "========================================"

    for config in "${MODELS_CONFIG[@]}"; do
        # 1. Split the config string "model:batch"
        IFS=':' read -r model_path batch_size <<< "$config"

        # 2. Apply Filter (if set)
        if [[ -n "$FILTER" && "$model_path" != *"$FILTER"* ]]; then
            continue
        fi

        echo ">> Model: $model_path | Batch: $batch_size"
        
        # 3. Execution
        # --- A. LogProb Experiment ---
        if [ "$RUN_LOGPROBS" = true ]; then
            script_path="./scripts/${script_prefix}_experiment.py"
            
            echo "   [LogProbs]   Running..."
            python "$script_path" \
                --model "$model_path" \
                --batch_size "$batch_size" \
                --device "$DEVICE" \
                $extra_args
            
            if [ $? -ne 0 ]; then echo "   !!!! LogProb Exp Failed for $model_path !!!!"; fi
        fi

        # --- B. MCQ Experiment ---
        if [ "$RUN_MCQ" = true ]; then
            script_path="./scripts/${script_prefix}_mcq_experiment.py"
            
            echo "   [MCQ]      Running..."
            python "$script_path" \
                --model "$model_path" \
                --batch_size "$batch_size" \
                --device "$DEVICE" \
                $extra_args

            if [ $? -ne 0 ]; then echo "   !!!! MCQ Exp Failed for $model_path !!!!"; fi
        fi

        echo "---------------------------------------------"
    done
    echo ""
}

# ==========================================
# EXECUTION
# ==========================================

# 1. LFW
run_dataset_experiments "LFW" "lfw" ""

# 2. AgeDB
run_dataset_experiments "AgeDB" "agedb" ""

# 3. CelebA
run_dataset_experiments "CelebA" "celeba" "--partition_num 2"

# 4. CASIA-Iris
run_dataset_experiments "CASIA-Iris" "casia" "--num_pairs 20000 --random_seed 42"

# 5. FVC
run_dataset_experiments "FVC" "fvc" ""

echo "All experiments completed."