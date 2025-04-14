#!/bin/bash

# This script sets up the environment for running the migraine prediction with proper library paths

# Define the paths to lapack libraries
LAPACK_PATH="/opt/homebrew/opt/lapack"

# Export environment variables needed for LAPACK detection
export LDFLAGS="-L$LAPACK_PATH/lib"
export CPPFLAGS="-I$LAPACK_PATH/include"

# Set DYLD_LIBRARY_PATH to include homebrew's LAPACK directory
export DYLD_LIBRARY_PATH="$LAPACK_PATH/lib:$DYLD_LIBRARY_PATH"

# Create symlinks for the specific libraries PyGMO is looking for
# This script can be run multiple times safely - it won't create duplicate symlinks
if [ ! -f "/usr/local/lib/liblapack.3.dylib" ]; then
    echo "Creating symlink for liblapack.3.dylib in /usr/local/lib"
    sudo mkdir -p /usr/local/lib
    sudo ln -sf "$LAPACK_PATH/lib/liblapack.dylib" "/usr/local/lib/liblapack.3.dylib"
fi

if [ ! -f "$LAPACK_PATH/lib/liblapack.3.dylib" ]; then
    echo "Creating symlink for liblapack.3.dylib in $LAPACK_PATH/lib"
    ln -sf "$LAPACK_PATH/lib/liblapack.dylib" "$LAPACK_PATH/lib/liblapack.3.dylib"
fi

# Install required Python packages if needed
pip install imblearn scikit-learn --quiet

# Run the migraine prediction script with the new data balancing parameters
echo "Running migraine prediction script with proper library paths..."
echo "================================================================"

# Parameters for the run
BALANCE_METHOD="smote"  # Options: smote, random_over, random_under, none
SAMPLING_RATIO="0.5"    # Target ratio for balancing

./src/scripts/run_migraine.sh \
    --balance_method $BALANCE_METHOD \
    --sampling_ratio $SAMPLING_RATIO \
    "$@"