#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status

# Set environment variables
export CUDA_VISIBLE_DEVICES=1

# Script directory and base path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
echo "Base directory: $BASE_DIR"

# Add BASE_DIR to PYTHONPATH so Python can find modules in src/
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Check for GPU availability
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "-1" ]; then
    CPU_FLAG="--cpu"
    echo "Running on CPU"
else
    CPU_FLAG=""
    echo "Running with GPU: $CUDA_VISIBLE_DEVICES"
fi

# Create data and output directories
DATA_DIR="$BASE_DIR/data/migraine"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

OUTPUT_DIR="$BASE_DIR/results/migraine"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Generate synthetic migraine dataset
echo "Generating synthetic dataset for Migraine Prediction..."
python "$SCRIPT_DIR/create_migraine_dataset.py" \
    --output_dir "$DATA_DIR" \
    --num_patients 100 \
    --days 30
echo "Migraine dataset generation complete."

# Step 2: Run the migraine prediction pipeline with PyGMO optimization
echo "Running migraine prediction with PyGMO-enhanced FuseMOE..."
python "$SCRIPT_DIR/run_migraine_prediction.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$BASE_DIR/cache/migraine" \
    --output_dir "$OUTPUT_DIR" \
    --start_date "2023-01-01" \
    --end_date "2023-01-30" \
    --use_pygmo \
    --expert_algorithm "de" \
    --gating_algorithm "pso" \
    --expert_population_size 10 \
    --gating_population_size 10 \
    --expert_generations 5 \
    --gating_generations 5 \
    --num_experts 8 \
    --modality_experts "eeg:3,weather:2,sleep:2,stress:1" \
    --hidden_size 64 \
    --top_k 2 \
    --patient_adaptation \
    --patient_iterations 3 \
    --cross_method "moe" \
    --gating_function "laplace" \
    --router_type "joint" \
    $CPU_FLAG

echo "Migraine prediction completed. Results saved to $OUTPUT_DIR"

# Optional: Run advanced patient-specific adaptation
if [ "$1" == "--patient-adaptation" ]; then
    PATIENT_ID="$2"
    if [ -z "$PATIENT_ID" ]; then
        echo "Error: Patient ID required for patient-specific adaptation."
        echo "Usage: ./run_migraine.sh --patient-adaptation [patient_id]"
        exit 1
    fi
    
    echo "Running patient-specific adaptation for patient $PATIENT_ID..."
    python "$SCRIPT_DIR/run_migraine_prediction.py" \
        --data_dir "$DATA_DIR" \
        --cache_dir "$BASE_DIR/cache/migraine" \
        --output_dir "$OUTPUT_DIR/patient_$PATIENT_ID" \
        --patient_id "$PATIENT_ID" \
        --use_pygmo \
        --expert_algorithm "de" \
        --gating_algorithm "pso" \
        --expert_population_size 5 \
        --gating_population_size 5 \
        --expert_generations 3 \
        --gating_generations 3 \
        --patient_adaptation \
        --patient_iterations 5 \
        --load_base_model "$OUTPUT_DIR/model.pth" \
        $CPU_FLAG
    
    echo "Patient-specific adaptation completed. Results saved to $OUTPUT_DIR/patient_$PATIENT_ID"
fi 