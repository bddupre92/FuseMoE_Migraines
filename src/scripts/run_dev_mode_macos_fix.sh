#!/bin/bash

# This script runs the migraine prediction pipeline in development mode
# with reduced dataset size and simplified parameters for faster execution
# Includes macOS-specific fixes for PyTorch library loading issues

# Calculate the correct script and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Base directory: $BASE_DIR"

# ====== macOS-specific fixes for PyTorch ======
# Unset PYTHONPATH to avoid conflicts
unset PYTHONPATH

# Add BASE_DIR to PYTHONPATH for module imports
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Disable MKL threading optimizations which can cause conflicts
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# macOS specific fix for PyTorch library loading issues
# This tells PyTorch to use its own BLAS implementation rather than system Accelerate
export PYTORCH_ENABLE_MKL_OPTIMIZATIONS=0

# Pointer to system libraries - important for macOS
export DYLD_FALLBACK_LIBRARY_PATH="/usr/lib:/usr/local/lib:$DYLD_FALLBACK_LIBRARY_PATH"

# ===== Activate Conda Environment =====
# Attempt to activate the fusemoe_env
# Check if conda is initialized
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fusemoe_env
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment 'fusemoe_env'."
        echo "Please ensure the environment exists and is properly configured."
        exit 1
    fi
    echo "Activated conda environment: fusemoe_env"
else
    echo "Error: Conda initialization script not found. Cannot activate environment."
    exit 1
fi

# Create data and output directories
DATA_DIR="$BASE_DIR/data/migraine_dev"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

OUTPUT_DIR="$BASE_DIR/results/migraine_dev"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# ===== IMPROVED DEVELOPMENT PARAMETERS =====
# Reduced data generation parameters for development
NUM_PATIENTS=25    # Down from 500
NUM_DAYS=10        # Down from 60
START_DATE_STR="2023-01-01"
END_DATE_STR=$(date -v+${NUM_DAYS}d -v-1d -j -f %Y-%m-%d "$START_DATE_STR" +%Y-%m-%d)

# Step 1: Generate smaller synthetic migraine dataset with better balance
echo "Generating small synthetic dataset for development with improved balance..."
python "$SCRIPT_DIR/create_migraine_dataset.py" \
    --output_dir "$DATA_DIR" \
    --num_patients $NUM_PATIENTS \
    --days $NUM_DAYS \
    --avg_migraine_freq 0.15 \
    --seed 42
echo "Development dataset generation complete."

# ===== Explicitly generate aligned weather data =====
echo "Generating aligned synthetic weather data..."
python "$SCRIPT_DIR/generate_weather_data.py" \
    --output_dir "$DATA_DIR/weather" \
    --start_date "$START_DATE_STR" \
    --end_date "$END_DATE_STR" \
    --latitude 40.7128 \
    --longitude -74.006
echo "Synthetic weather data generation complete."
# ==================================================

# Step 2: Run a simplified version of the migraine prediction pipeline with more epochs & early stopping
echo "Running simplified migraine prediction for development..."
python -W ignore "$SCRIPT_DIR/run_migraine_prediction.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$BASE_DIR/cache/migraine_dev" \
    --output_dir "$BASE_DIR/results/migraine_dev_none" \
    --start_date "$START_DATE_STR" \
    --end_date "$END_DATE_STR" \
    --num_experts 8 \
    --modality_experts "eeg:3,weather:2,sleep:2,stress:1" \
    --hidden_size 64 \
    --top_k 2 \
    --window_size 12 \
    --prediction_horizon 6 \
    --cv 3 \
    --cv_strategy "stratifiedgroupkfold" \
    --balance_method "smote" \
    --sampling_ratio 0.6 \
    --class_weight "balanced" \
    --threshold_search \
    --optimize_metric "f1" \
    --dropout_rate 0.25 \
    --imputation_method "none" \
    --dev_mode \
    --dev_epochs 15 \
    --dev_batch_size 8 \
    --use_pygmo \
    --expert_population_size 10 \
    --gating_population_size 10 \
    --expert_generations 5 \
    --gating_generations 5 \
    --cpu

echo "Development run completed. Results saved to $OUTPUT_DIR"