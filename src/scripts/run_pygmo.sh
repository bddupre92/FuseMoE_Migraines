#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Script directory and base path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
echo "Base directory: $BASE_DIR"

# Check for GPU availability
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "-1" ]; then
    CPU_FLAG="--cpu"
    echo "Running on CPU"
else
    CPU_FLAG=""
    echo "Running with GPU: $CUDA_VISIBLE_DEVICES"
fi

# Create data directory if needed
DATA_DIR="$BASE_DIR/Data/ihm"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

# Create output directory
OUTPUT_DIR="$BASE_DIR/run/PyGMO_TS"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Generate synthetic dataset (if it doesn't already exist)
if [ ! -f "$DATA_DIR/trainp2x_data.pkl" ]; then
    echo "Generating synthetic dataset for FuseMOE..."
    python "$SCRIPT_DIR/create_minimal_dataset.py"
    echo "Dataset generation complete."
fi

# Step 2: Run the main.py script with PyGMO-enhanced FuseMOE parameters
echo "Running main.py with PyGMO-enhanced FuseMOE parameters..."
python -W ignore "$SCRIPT_DIR/main.py" \
    --num_train_epochs 6 \
    --modeltype 'TS' \
    --kernel_size 1 \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --seed 42 \
    --gradient_accumulation_steps 16 \
    --num_update_bert_epochs 2 \
    --bertcount 3 \
    --ts_learning_rate 0.0004 \
    --notes_order 'Last' \
    --num_of_notes 5 \
    --max_length 1024 \
    --layers 3 \
    --output_dir "$OUTPUT_DIR" \
    --embed_dim 128 \
    --model_name "bioLongformer" \
    --task 'ihm' \
    --file_path "$DATA_DIR" \
    --num_labels 2 \
    --num_heads 8 \
    --embed_time 64 \
    --tt_max 48 \
    --mixup_level 'batch' \
    --cross_method "moe" \
    --gating_function "laplace" \
    --num_of_experts 3 \
    --top_k 2 \
    --router_type 'joint' \
    --use_pygmo \
    --expert_algorithm "de" \
    --gating_algorithm "pso" \
    --expert_population_size 10 \
    --gating_population_size 10 \
    --expert_generations 5 \
    --gating_generations 5 \
    $CPU_FLAG \
    --reg_ts

echo "PyGMO-enhanced FuseMOE run completed. Results saved to $OUTPUT_DIR" 