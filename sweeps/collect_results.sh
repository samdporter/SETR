#!/bin/bash

# --- SETR Sweep Results Collection Script ---
# Gathers completed results from individual job output directories into a summary CSV
# Usage: ./collect_results.sh <sweep_name>

# Auto-detect base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SWEEP_OUTPUT_DIR="$SCRIPT_DIR/output"

SWEEP_NAME="$1"

if [ -z "$SWEEP_NAME" ]; then
    echo "=== SETR Results Collection ==="
    echo "Usage: $0 <sweep_name>"
    echo ""
    echo "Available sweeps:"
    ls -1 $SWEEP_OUTPUT_DIR/ 2>/dev/null | sed 's/^/  /' || echo "  No sweeps found"
    exit 1
fi

if [ ! -d "$SWEEP_OUTPUT_DIR/$SWEEP_NAME" ]; then
    echo "Error: Sweep directory not found: $SWEEP_OUTPUT_DIR/$SWEEP_NAME"
    exit 1
fi

echo "=== Collecting Results: $SWEEP_NAME ==="
echo "Scanning directory: $SWEEP_OUTPUT_DIR/$SWEEP_NAME"
echo ""

RESULTS_FILE="$SWEEP_OUTPUT_DIR/${SWEEP_NAME}_summary.csv"

# Create CSV header
echo "alpha,beta,status,final_objective,num_iterations,output_dir,completion_time" > "$RESULTS_FILE"

completed_count=0
failed_count=0
pending_count=0

# Process each alpha/beta combination directory
for dir in "$SWEEP_OUTPUT_DIR/$SWEEP_NAME"/alpha_*_beta_*; do
    if [ -d "$dir" ]; then
        # Extract alpha and beta from directory name
        basename_dir=$(basename "$dir")
        alpha=$(echo "$basename_dir" | sed 's/alpha_\([^_]*\)_beta_.*/\1/')
        beta=$(echo "$basename_dir" | sed 's/.*_beta_\([^_]*\)/\1/')
        
        # Check job completion status
        if [ -f "$dir/job_completion.txt" ]; then
            status=$(grep "status=" "$dir/job_completion.txt" | cut -d'=' -f2 | cut -d',' -f1)
            completion_time=$(grep "end_time=" "$dir/job_completion.txt" | cut -d'=' -f3-)
            if [ "$status" = "completed" ]; then
                ((completed_count++))
            else
                ((failed_count++))
            fi
        else
            status="pending"
            completion_time="N/A"
            ((pending_count++))
        fi
        
        # Extract final objective value from BSREM objective file
        final_objective="N/A"
        num_iterations="N/A"
        
        # Look for objective CSV files (pattern: bsrem_objective_a_*_b_*.csv)
        objective_file=$(find "$dir" -name "bsrem_objective_a_${alpha}_b_${beta}.csv" | head -1)
        if [ ! -f "$objective_file" ]; then
            # Fallback: look for any bsrem_objective file
            objective_file=$(find "$dir" -name "bsrem_objective_*.csv" | head -1)
        fi
        
        if [ -f "$objective_file" ]; then
            # Get the last (final) objective value and count iterations
            final_line=$(tail -1 "$objective_file" 2>/dev/null)
            if [ -n "$final_line" ] && [ "$final_line" != "0" ]; then
                final_objective="$final_line"
                num_iterations=$(wc -l < "$objective_file" 2>/dev/null)
                # Subtract 1 if there's a header
                if head -1 "$objective_file" | grep -q "[a-zA-Z]"; then
                    num_iterations=$((num_iterations - 1))
                fi
            fi
        fi
        
        # Add row to CSV
        echo "$alpha,$beta,$status,$final_objective,$num_iterations,$dir,$completion_time" >> "$RESULTS_FILE"
    fi
done

echo "Results Summary:"
echo "  Completed: $completed_count"
echo "  Failed: $failed_count" 
echo "  Pending: $pending_count"
echo "  Total combinations: $((completed_count + failed_count + pending_count))"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

if [ $completed_count -gt 0 ]; then
    echo "Completed jobs with best (lowest) objective values:"
    # Show top 5 best results (lowest objective values)
    tail -n +2 "$RESULTS_FILE" | grep "completed" | sort -t',' -k4 -n | head -5 | \
    while IFS=',' read -r alpha beta status obj iter dir time; do
        echo "  α=$alpha, β=$beta: objective=$obj (${iter} iterations)"
    done
fi

echo ""
echo "To view full results:"
echo "  cat $RESULTS_FILE"
echo "  column -t -s',' $RESULTS_FILE | less"