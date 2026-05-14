#!/bin/bash

# 1. Kill any zombie processes from previous failed runs
killall -9 python 2>/dev/null

# 2. Critical Resource Management
# We MUST prevent JAX from seeing the whole 141GB.
# 141GB / 20 processes = ~7GB. 
# We set a hard limit of 6GB per process to be safe.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.04 

# 3. Create Logs Directory
mkdir -p logs/output_files
rm -f logs/output_files/*.log

PARAMS=("0.000" "0.010" "0.020" "0.030" "0.040" "0.050" "0.060" "0.070" "0.080" "0.090" "0.100" "0.110" "0.120" "0.130" "0.140" "0.150" "0.160" "0.170" "0.180" "0.190")

echo "Launching 20 training processes on H200 (Default Multi-Process Mode)..."

for p in "${PARAMS[@]}"
do
   # Use 'stdbuf' to ensure Python doesn't buffer the output
   # so your log monitoring is actually 'live'
   stdbuf -oL python singleTraining.py --param $p > logs/output_files/train_$p.log 2>&1 &
   echo "Started Param $p"
   sleep 2 # Important: Staggering allows the GPU to allocate memory chunks one-by-one
done

# --- Monitor Loop ---
echo "Monitoring training (Updates every 30s)..."
while true; do
    RUNNING_COUNT=$(jobs -r | wc -l)
    if [ "$RUNNING_COUNT" -eq 0 ]; then break; fi

    echo "--- $(date +%H:%M:%S) | Active: $RUNNING_COUNT ---"
    for log in logs/output_files/*.log; do
        PARAM_VAL=$(basename "$log" .log | cut -d'_' -f2)
        LAST_LINE=$(tail -n 1 "$log" | cut -c 1-80) # Truncate for clean table
        echo "[$PARAM_VAL]: $LAST_LINE"
    done
    sleep 30
done

wait
echo "All training finished."
