#!/bin/bash
# Define start time for execution
start_time=$(date +%s%N)

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to execute each step with error handling
execute_step() {
    echo "Executing: $1"
    if python3 "$1"; then
        echo -e "${GREEN}Step completed successfully.${NC}"
    else
        echo -e "${RED}Error executing $1. Exiting...${NC}"
        exit 1
    fi
    echo ""
}

# Enable immediate exit on error
set -e

# Run each step
echo -e "\033[94mStart Download BCause Dataset...\033[0m"
execute_step "0_download_bcause.py"
echo -e "\033[94mStart Clustering Pipeline...\033[0m"
execute_step "1_clustering.py"
echo -e "\033[94mStart text2KG Pipeline...\033[0m"
execute_step "2_text2KG.py"

# Calculate elapsed time
end_time=$(date +%s%N)
elapsed_time=$((end_time - start_time))
elapsed_seconds=$((elapsed_time / 1000000000))
elapsed_minutes=$((elapsed_seconds / 60))
remaining_seconds=$((elapsed_seconds % 60))
milliseconds=$((elapsed_time / 1000000 % 1000))

# Output elapsed time
echo "Elapsed time: $elapsed_minutes minutes, $remaining_seconds seconds, $milliseconds milliseconds"