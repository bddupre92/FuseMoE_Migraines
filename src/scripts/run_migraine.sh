#!/bin/bash

# Improved conda environment activation that's more robust on macOS
echo "Activating conda environment: fusemoe_env"

# Try multiple methods to activate the conda environment
# Method 1: Initialize conda first if needed
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Initializing conda using conda.sh"
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate fusemoe_env || {
        echo "Method 1 failed, trying alternative methods..."
    }
else
    echo "Could not find conda.sh, trying alternative methods..."
fi

# Method 2: Direct environment PATH setting (fallback)
if [[ "$CONDA_DEFAULT_ENV" != "fusemoe_env" ]]; then
    echo "Setting environment variables directly..."
    if [ -d "/opt/anaconda3/envs/fusemoe_env/bin" ]; then
        export PATH="/opt/anaconda3/envs/fusemoe_env/bin:$PATH"
        echo "Path updated to use fusemoe_env binaries"
    fi
fi

# Method 3: Try old-style activation as a last resort
if [[ "$CONDA_DEFAULT_ENV" != "fusemoe_env" ]]; then
    echo "Trying source activate as a last resort..."
    source activate fusemoe_env || {
        echo "WARNING: All activation methods failed. Proceeding with current environment."
    }
fi

echo "Proceeding with environment: $CONDA_DEFAULT_ENV"

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

# Define data generation parameters
NUM_PATIENTS=500
NUM_DAYS=60
START_DATE_STR="2023-01-01"

# Set default cross-validation parameters
CV_FOLDS=5
CV_STRATEGY="stratified"
CV_SHUFFLE="--cv_shuffle"  # Enable shuffling by default

# Set default data balancing parameters
BALANCE_METHOD="smote"
SAMPLING_RATIO="0.8"  # Increased from 0.5 for better balance

# Process command-line arguments for cross-validation and data balancing
while [[ $# -gt 0 ]]; do
  case $1 in
    --cv)
      CV_FOLDS="$2"
      shift 2
      ;;
    --cv-strategy)
      CV_STRATEGY="$2"
      shift 2
      ;;
    --cv-shuffle)
      CV_SHUFFLE="--cv_shuffle"
      shift
      ;;
    --patient-adaptation)
      PATIENT_ADAPTATION=true
      PATIENT_ID="$2"
      shift 2
      ;;
    --balance_method)
      BALANCE_METHOD="$2"
      shift 2
      ;;
    --sampling_ratio)
      SAMPLING_RATIO="$2"
      shift 2
      ;;
    *)
      # Skip unknown arguments
      shift
      ;;
  esac
done

# Calculate end date based on start date and number of days
END_DATE_STR=$(date -v+${NUM_DAYS}d -v-1d -j -f %Y-%m-%d "$START_DATE_STR" +%Y-%m-%d)

# Step 1: Generate synthetic migraine dataset with improved class balance
echo "Generating synthetic dataset for Migraine Prediction..."
python "$SCRIPT_DIR/create_migraine_dataset.py" \
    --output_dir "$DATA_DIR" \
    --num_patients $NUM_PATIENTS \
    --days $NUM_DAYS \
    --avg_migraine_freq 0.2  # Lower frequency for better initial balance
echo "Migraine dataset generation complete."

# Step 2: Run the migraine prediction pipeline with cross-validation
echo "Running migraine prediction with FuseMOE and ${CV_FOLDS}-fold cross-validation (${CV_STRATEGY})..."
python "$SCRIPT_DIR/run_migraine_prediction.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$BASE_DIR/cache/migraine" \
    --output_dir "$OUTPUT_DIR" \
    --start_date "$START_DATE_STR" \
    --end_date "$END_DATE_STR" \
    --expert_algorithm "de" \
    --gating_algorithm "pso" \
    --expert_population_size 20 \
    --gating_population_size 20 \
    --expert_generations 20 \
    --gating_generations 20 \
    --num_experts 8 \
    --modality_experts "eeg:3,weather:2,sleep:2,stress:1" \
    --hidden_size 128 \
    --top_k 2 \
    --cross_method "moe" \
    --gating_function "laplace" \
    --router_type "joint" \
    --imputation_method "knn" \
    --cv $CV_FOLDS \
    --cv_strategy "stratified" \
    $CV_SHUFFLE \
    --balance_method "smote" \
    --sampling_ratio "0.8" \
    --class_weight "balanced" \
    --threshold_search \
    --optimize_metric "balanced_accuracy" \
    --use_pygmo \
    --batch_size 32 \
    $CPU_FLAG

echo "Migraine prediction with cross-validation completed. Results saved to $OUTPUT_DIR"

# Optional: Run advanced patient-specific adaptation
if [ "$PATIENT_ADAPTATION" = true ]; then
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
        --expert_algorithm "de" \
        --gating_algorithm "pso" \
        --expert_population_size 5 \
        --gating_population_size 5 \
        --expert_generations 3 \
        --gating_generations 3 \
        --patient_adaptation \
        --patient_iterations 5 \
        --load_base_model "$OUTPUT_DIR/best_model.pth" \
        --cv 1 \
        --balance_method "$BALANCE_METHOD" \
        --sampling_ratio "$SAMPLING_RATIO" \
        --class_weight "balanced" \
        --threshold_search \
        --optimize_metric "balanced_accuracy" \
        --batch_size 32 \
        $CPU_FLAG
    
    echo "Patient-specific adaptation completed. Results saved to $OUTPUT_DIR/patient_$PATIENT_ID"
fi