#!/bin/bash

# This script runs the migraine prediction pipeline in development mode
# with reduced dataset size and simplified parameters for faster execution

# Define the paths to lapack libraries (same as run_with_lapack.sh)
LAPACK_PATH="/opt/homebrew/opt/lapack"
export LDFLAGS="-L$LAPACK_PATH/lib"
export CPPFLAGS="-I$LAPACK_PATH/include"
export DYLD_LIBRARY_PATH="$LAPACK_PATH/lib:$DYLD_LIBRARY_PATH"

# Create symlinks for the specific libraries PyGMO is looking for
if [ ! -f "/usr/local/lib/liblapack.3.dylib" ]; then
    echo "Creating symlink for liblapack.3.dylib in /usr/local/lib"
    sudo mkdir -p /usr/local/lib
    sudo ln -sf "$LAPACK_PATH/lib/liblapack.dylib" "/usr/local/lib/liblapack.3.dylib"
fi

if [ ! -f "$LAPACK_PATH/lib/liblapack.3.dylib" ]; then
    echo "Creating symlink for liblapack.3.dylib in $LAPACK_PATH/lib"
    ln -sf "$LAPACK_PATH/lib/liblapack.dylib" "$LAPACK_PATH/lib/liblapack.3.dylib"
fi

# Calculate the correct script and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Base directory: $BASE_DIR"

# Add BASE_DIR to PYTHONPATH so Python can find modules in src/
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Create data and output directories
DATA_DIR="$BASE_DIR/data/migraine_dev"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

OUTPUT_DIR="$BASE_DIR/results/migraine_dev"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Reduced data generation parameters for development
NUM_PATIENTS=25    # Down from 500
NUM_DAYS=10        # Down from 60
START_DATE_STR="2023-01-01"
END_DATE_STR=$(date -v+${NUM_DAYS}d -v-1d -j -f %Y-%m-%d "$START_DATE_STR" +%Y-%m-%d)

# Step 1: Generate smaller synthetic migraine dataset
echo "Generating small synthetic dataset for development..."
python "$SCRIPT_DIR/create_migraine_dataset.py" \
    --output_dir "$DATA_DIR" \
    --num_patients $NUM_PATIENTS \
    --days $NUM_DAYS \
    --seed 42
echo "Development dataset generation complete."

# Step 2: Run a simplified version of the migraine prediction pipeline
echo "Running simplified migraine prediction for development..."
python "$SCRIPT_DIR/run_migraine_prediction.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$BASE_DIR/cache/migraine_dev" \
    --output_dir "$OUTPUT_DIR" \
    --start_date "$START_DATE_STR" \
    --end_date "$END_DATE_STR" \
    --num_experts 4 \
    --modality_experts "eeg:2,weather:1,sleep:1,stress:0" \
    --hidden_size 32 \
    --top_k 1 \
    --window_size 8 \
    --cv 2 \
    --cv_strategy "kfold" \
    --balance_method "random_over" \
    --sampling_ratio 0.5 \
    --dev_mode \
    --dev_epochs 2 \
    --dev_batch_size 4 \
    --skip_visualizations \
    --cpu

echo "Development run completed. Results saved to $OUTPUT_DIR"