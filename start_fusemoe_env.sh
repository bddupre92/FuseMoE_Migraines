#!/bin/bash
# This script activates the fusemoe_env conda environment
# and navigates to the scripts directory

# Activate the fusemoe_env environment
conda activate fusemoe_env

# Navigate to the scripts directory
cd src/scripts

echo "Activated fusemoe_env environment and navigated to scripts directory"
echo "Run the demo with: python enhanced_pygmo_fusemoe_demo.py --demo migraine" 