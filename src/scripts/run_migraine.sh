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
export DATA_DIR="$BASE_DIR/data/migraine"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

export RESULTS_DIR="$BASE_DIR/results/migraine"
mkdir -p "$RESULTS_DIR"
echo "Output directory: $RESULTS_DIR"

# --- Default Parameters for Full Run ---
export NUM_PATIENTS=500
export NUM_DAYS=60
export AVG_MIGRAINE_FREQ=0.2
export CV_FOLDS=5
export EXPERT_POP_SIZE=20
export GATING_POP_SIZE=20
export EXPERT_GENS=20
export GATING_GENS=20
export DEV_MODE_FLAG="" # Empty by default
export SEED=42 # Define SEED variable explicitly here
export WINDOW_SIZE_PARAM=24 # Default window size (Changed from 12)
export PREDICTION_HORIZON_PARAM=6 # Default horizon

# Set default cross-validation parameters
export CV_STRATEGY="groupkfold" # Options: "groupkfold", "stratifiedkfold"
export CV_SHUFFLE="--cv_shuffle"  # Enable shuffling by default

# Set default data balancing parameters
export BALANCE_METHOD="smote"
export SAMPLING_RATIO="0.6" # Target ratio for SMOTE/random sampling
export CLASS_WEIGHT="balanced" # Options: "balanced", "none"
export N_SPLITS=5

# --- Parse Command-Line Arguments ---
DEV_MODE_ENABLED=false
# Initialize python script specific args
PYTHON_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev)
      DEV_MODE_ENABLED=true
      echo ">>> DEVELOPMENT MODE ENABLED <<<"
      PYTHON_ARGS+=" --dev_mode"
      shift # past argument
      ;;
    --cv)
      CV_FOLDS="$2"
      PYTHON_ARGS+=" --cv $2"
      shift 2
      ;;
    --cv-strategy)
      CV_STRATEGY="$2"
      PYTHON_ARGS+=" --cv_strategy $2"
      shift 2
      ;;
    --cv-shuffle)
      # CV_SHUFFLE="--cv_shuffle" # This is default now, flag adds it to PYTHON_ARGS
      PYTHON_ARGS+=" --cv_shuffle"
      shift
      ;;
    --window_size)
       WINDOW_SIZE_PARAM="$2"
       PYTHON_ARGS+=" --window_size $2"
       shift 2
       ;;
    --prediction_horizon)
       PREDICTION_HORIZON_PARAM="$2"
       PYTHON_ARGS+=" --prediction_horizon $2"
       shift 2
       ;;
    --patient-adaptation)
      # This doesn't seem to be fully used/passed currently, placeholder
      PATIENT_ADAPTATION=true 
      PATIENT_ID="$2"
      PYTHON_ARGS+=" --patient_adaptation --patient_id $2" # Pass to python
      shift 2
      ;;
    --balance_method)
      BALANCE_METHOD="$2"
      PYTHON_ARGS+=" --balance_method $2"
      shift 2
      ;;
    --sampling_ratio)
      SAMPLING_RATIO="$2"
      PYTHON_ARGS+=" --sampling_ratio $2"
      shift 2
      ;;
    # Add cases for other parameters passed directly to python script if needed
    # Example: --hidden_size, --learning_rate etc.
    *)
      # Assume unknown args are for the python script
      PYTHON_ARGS+=" $1"
      if [[ $# -gt 1 ]] && [[ $2 != --* ]]; then # Check if next arg is a value
          PYTHON_ARGS+=" $2"
          shift 2
      else 
          shift # Treat as a flag
      fi
      ;;
  esac
done

# --- Apply Dev Mode Overrides if Enabled ---
if [ "$DEV_MODE_ENABLED" = true ]; then
  echo "Applying development mode SHELL parameter overrides..."
  NUM_PATIENTS=25    # Reduced patients
  NUM_DAYS=10        # Reduced days
  AVG_MIGRAINE_FREQ=0.15 # Slightly different balance for dev
  # --- Keep CV folds override for shell script logic if needed elsewhere ---
  # CV_FOLDS=3         # Fewer folds 
  # --- Keep PyGMO overrides for shell script logic if needed elsewhere ---
  # EXPERT_POP_SIZE=10 # Smaller PyGMO pop
  # GATING_POP_SIZE=10
  # EXPERT_GENS=5      # Fewer PyGMO gens
  # GATING_GENS=5
  
  # --- Dev defaults for python script args (added to PYTHON_ARGS if not already set) ---
  # Check if specific args were already provided before adding dev defaults
  if [[ ! "$PYTHON_ARGS" =~ "--cv " ]]; then PYTHON_ARGS+=" --cv 3"; fi
  if [[ ! "$PYTHON_ARGS" =~ "--expert_population_size" ]]; then PYTHON_ARGS+=" --expert_population_size 10"; fi
  if [[ ! "$PYTHON_ARGS" =~ "--gating_population_size" ]]; then PYTHON_ARGS+=" --gating_population_size 10"; fi
  if [[ ! "$PYTHON_ARGS" =~ "--expert_generations" ]]; then PYTHON_ARGS+=" --expert_generations 5"; fi
  if [[ ! "$PYTHON_ARGS" =~ "--gating_generations" ]]; then PYTHON_ARGS+=" --gating_generations 5"; fi
  # The --dev_mode flag itself is already added
  # Window size and prediction horizon are handled by user flags or script defaults now.
fi

# --- Set final Python script arguments based on defaults and flags ---
# Ensure core defaults are present if not overridden by flags
if [[ ! "$PYTHON_ARGS" =~ "--window_size" ]]; then PYTHON_ARGS+=" --window_size $WINDOW_SIZE_PARAM"; fi
if [[ ! "$PYTHON_ARGS" =~ "--prediction_horizon" ]]; then PYTHON_ARGS+=" --prediction_horizon $PREDICTION_HORIZON_PARAM"; fi
if [[ ! "$PYTHON_ARGS" =~ "--sampling_ratio" ]]; then PYTHON_ARGS+=" --sampling_ratio $SAMPLING_RATIO"; fi
if [[ ! "$PYTHON_ARGS" =~ "--balance_method" ]]; then PYTHON_ARGS+=" --balance_method $BALANCE_METHOD"; fi
if [[ ! "$PYTHON_ARGS" =~ "--cv_strategy" ]]; then PYTHON_ARGS+=" --cv_strategy $CV_STRATEGY"; fi
if [[ ! "$PYTHON_ARGS" =~ "--cv " ]] && [[ $CV_FOLDS -gt 0 ]]; then PYTHON_ARGS+=" --cv $CV_FOLDS"; fi # Use shell default if no flag
if [[ "$CV_SHUFFLE" = "--cv_shuffle" ]] && [[ ! "$PYTHON_ARGS" =~ "--cv_shuffle" ]]; then PYTHON_ARGS+=" --cv_shuffle"; fi

# Add remaining necessary defaults if not set
if [[ ! "$PYTHON_ARGS" =~ "--class_weight" ]]; then PYTHON_ARGS+=" --class_weight $CLASS_WEIGHT"; fi
if [[ ! "$PYTHON_ARGS" =~ "--optimize_metric" ]]; then PYTHON_ARGS+=" --optimize_metric f1"; fi # Defaulting to f1 now
if [[ ! "$PYTHON_ARGS" =~ "--threshold_search" ]]; then PYTHON_ARGS+=" --threshold_search"; fi 
if [[ ! "$PYTHON_ARGS" =~ "--hidden_size" ]]; then PYTHON_ARGS+=" --hidden_size 128"; fi 
# Add other necessary defaults here...

# Add CPU flag if needed
PYTHON_ARGS+=" $CPU_FLAG"

# Add seed
PYTHON_ARGS+=" --seed $SEED"

# --- Derived Parameters ---
START_DATE_STR="2023-01-01"
# Calculate end date based on start date and number of days
END_DATE_STR=$(date -v+${NUM_DAYS}d -v-1d -j -f %Y-%m-%d "$START_DATE_STR" +%Y-%m-%d)

# === Step 1: Generate synthetic migraine dataset ===
echo "Generating synthetic dataset for Migraine Prediction..."
echo "  Patients: $NUM_PATIENTS, Days: $NUM_DAYS, Avg Freq: $AVG_MIGRAINE_FREQ"
python "$SCRIPT_DIR/create_migraine_dataset.py" \
    --output_dir "$DATA_DIR" \
    --num_patients $NUM_PATIENTS \
    --days $NUM_DAYS \
    --avg_migraine_freq $AVG_MIGRAINE_FREQ \
    --seed $SEED
echo "Migraine dataset generation complete."

# === Step 2: Run the migraine prediction pipeline ===
echo "Running migraine prediction with FuseMOE..."
echo "  Using effective Python arguments: $PYTHON_ARGS"
# --- Call python script with the collected arguments --- 
python "$SCRIPT_DIR/run_migraine_prediction.py" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$BASE_DIR/cache/migraine" \
    --output_dir "$RESULTS_DIR" \
    --start_date "$START_DATE_STR" \
    --end_date "$END_DATE_STR" \
    $PYTHON_ARGS # Pass all collected args here
# --- --------------------------------------------- --- 

echo "Migraine prediction with cross-validation completed. Results saved to $RESULTS_DIR"

# === Optional: Run advanced patient-specific adaptation ===
# Note: Dev mode logic is NOT added here, as adaptation is usually specific
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
        --output_dir "$RESULTS_DIR/patient_$PATIENT_ID" \
        --patient_id "$PATIENT_ID" \
        --expert_algorithm "de" \
        --gating_algorithm "pso" \
        --expert_population_size 5 \
        --gating_population_size 5 \
        --expert_generations 3 \
        --gating_generations 3 \
        --patient_adaptation \
        --patient_iterations 5 \
        --load_base_model "$RESULTS_DIR/best_model.pth" \
        --cv 1 \
        --threshold_search \
        --optimize_metric "balanced_accuracy" \
        --batch_size 32 \
        $CPU_FLAG
    
    echo "Patient-specific adaptation completed. Results saved to $RESULTS_DIR/patient_$PATIENT_ID"
fi

echo "Script finished."