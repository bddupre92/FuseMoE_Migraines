# %% [markdown]
# Optimization Progress Visualization
#
# Objective: Load the detailed optimization history saved by the PyGMO-enhanced
# FuseMoE optimization process and visualize the convergence of fitness scores
# and other metrics for expert evolution and gating PSO.

# %% [markdown]
# ## 1. Setup and Imports

# %% 
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Imports and Logging Setup Complete")

# %% [markdown]
# ## 2. Configuration and Constants

# %% 
# Constants
HISTORY_FILENAME = "optimization_history.json"
RESULTS_DIR_NAME = "results/migraine"

print(f"Constants defined: HISTORY_FILENAME='{HISTORY_FILENAME}', RESULTS_DIR_NAME='{RESULTS_DIR_NAME}'")

# %% [markdown]
# ## 3. Utility Functions

# %% 
def find_project_root(marker_file='.git', start_dir=None):
    """Find the project root directory by searching upwards for a marker.
    Args:
        marker_file (str): A file or directory that marks the root.
        start_dir (str): The directory to start searching from.
    Returns:
        str: The absolute path to the project root, or None if not found.
    """
    if start_dir is None:
        try:
            # Assumes the script/notebook is in a subdirectory of the project
            start_dir = os.path.abspath(os.path.dirname(__file__))
        except NameError:
             # Fallback for interactive execution (like Jupyter)
             start_dir = os.getcwd()
        
        # If running from notebook, __file__ might not be defined directly
        if not os.path.exists(start_dir) or "/." in start_dir: # Crude check for temp notebook dirs
             start_dir = os.getcwd()

    current_dir = start_dir
    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # Reached the filesystem root
            # Fallback: Check if cwd contains the marker (for cases where script is run from root)
            if os.path.exists(os.path.join(os.getcwd(), marker_file)):
                return os.getcwd()
            return None
        current_dir = parent_dir

print("find_project_root function defined.")

# %% 
def plot_optimization_stage(history_data, stage_name, algorithm, output_dir):
    """Plots the convergence history for a single optimization stage.

    Args:
        history_data (list): List of dictionaries from the history file for the stage.
        stage_name (str): Name of the optimization stage (e.g., 'expert_evolution').
        algorithm (str): Name of the algorithm used for this stage.
        output_dir (str): Directory to save the plot.

    Returns:
        tuple: (best_fitness, eval_count_at_best) or (None, None) if plotting fails.
    """
    if not history_data:
        logging.warning(f"No history data found for stage '{stage_name}'. Skipping plot.")
        return None, None # Return None tuple if no data

    try:
        df = pd.DataFrame(history_data)
    except Exception as e:
        logging.error(f"Failed to create DataFrame for stage '{stage_name}': {e}")
        return None, None

    # Check for required columns used for plotting
    # Allow for different names for evaluation count (e.g., 'eval_count', 'fevals', 'Fevals')
    eval_col_options = ['eval_count', 'fevals', 'Fevals', 'generation', 'Gen']
    eval_col = next((col for col in eval_col_options if col in df.columns), None)
    
    # Allow for different names for best fitness (e.g., 'best_fitness', 'gbest')
    fitness_col_options = ['best_fitness', 'gbest']
    fitness_col = next((col for col in fitness_col_options if col in df.columns), None)

    if not eval_col:
        logging.warning(f"History data for '{stage_name}' is missing an evaluation count column (checked: {eval_col_options}). Skipping plot.")
        return None, None
    if not fitness_col:
        logging.warning(f"History data for '{stage_name}' is missing a fitness column (checked: {fitness_col_options}). Skipping plot.")
        return None, None

    # Determine if lower fitness is better (heuristic: check if values are typically negative or positive)
    is_minimization = df[fitness_col].mean() < 0 if pd.api.types.is_numeric_dtype(df[fitness_col]) else True # Default assume minimization
    
    # Find the best fitness and corresponding evaluation count
    if is_minimization:
        best_fitness_idx = df[fitness_col].idxmin()
    else:
        best_fitness_idx = df[fitness_col].idxmax()
        
    best_fitness_value = df.loc[best_fitness_idx, fitness_col]
    eval_count_at_best = df.loc[best_fitness_idx, eval_col]


    # Determine metrics to plot besides fitness and eval count
    metrics_to_plot = [col for col in df.columns if col not in [eval_col, fitness_col]]

    num_metrics = len(metrics_to_plot)
    # Adjust figure size based on the number of metrics
    fig_height = 4 + num_metrics * 3 # Base height + height per metric plot
    fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(12, fig_height), sharex=True)
    
    # Ensure axes is always iterable, even if only one plot
    if num_metrics == 0:
        # Check if 'axes' is already an array or Axes object
        if isinstance(axes, np.ndarray):
             axes = axes.flatten() # Flatten if it's already an array (e.g., from subplots)
        else:
            axes = [axes] # Make it a list containing the single Axes object
    else:
        axes = axes.flatten()

    sns.set_theme(style="whitegrid", palette="muted") # Use a slightly different theme

    # Define a helper for nicer metric names
    def get_metric_display_name(metric_key):
        name_map = {
            'best_fitness': 'Best Fitness',
            'gbest': 'Global Best Fitness',
            'eval_count': 'Evaluation Count',
            'fevals': 'Function Evaluations',
            'Fevals': 'Function Evaluations',
            'generation': 'Generation',
            'Gen': 'Generation',
            'dx': 'Parameter Convergence (dx)',
            'df': 'Fitness Convergence (df)',
            'Mean Vel.': 'Mean Velocity (PSO)',
            'Mean lbest': 'Mean Local Best (PSO)',
            'Avg. Dist.': 'Average Distance (PSO)',
            # Add more mappings as needed
        }
        return name_map.get(metric_key, metric_key.replace("_", " ").title())

    # Plot primary fitness
    fitness_label = get_metric_display_name(fitness_col)
    eval_label = get_metric_display_name(eval_col)
    fitness_ylabel = f"{fitness_label} ({'Lower' if is_minimization else 'Higher'} is Better)"
    
    sns.lineplot(ax=axes[0], data=df, x=eval_col, y=fitness_col, marker='o', markersize=4, label=fitness_label, color='#1f77b4', zorder=2)
    # Highlight the best point
    axes[0].scatter(eval_count_at_best, best_fitness_value, color='red', s=100, zorder=3, marker='*', label=f'Best: {best_fitness_value:.4f}')
    axes[0].set_title(f'{stage_name.replace("_", " ").title()} Convergence (Algorithm: {algorithm.upper()})', fontsize=16, pad=20)
    axes[0].set_ylabel(fitness_ylabel, fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot auxiliary metrics
    color_palette = sns.color_palette("viridis", num_metrics)
    for i, metric in enumerate(metrics_to_plot):
        ax_idx = i + 1
        metric_display_name = get_metric_display_name(metric)
        # Check if metric data is numeric before plotting
        if pd.api.types.is_numeric_dtype(df[metric]):
            sns.lineplot(ax=axes[ax_idx], data=df, x=eval_col, y=metric, marker='.', markersize=5, label=metric_display_name, color=color_palette[i])
            axes[ax_idx].set_ylabel(metric_display_name, fontsize=12)
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
        else:
             logging.warning(f"Metric '{metric}' in stage '{stage_name}' is not numeric. Skipping plot.")
             # Optionally hide the non-numeric plot axis
             axes[ax_idx].set_visible(False)


    axes[-1].set_xlabel(eval_label, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout further
    plot_filename = os.path.join(output_dir, f'{stage_name}_convergence.png')
    
    try:
        plt.savefig(plot_filename, dpi=150) # Increase DPI slightly
        logging.info(f"Saved {stage_name} convergence plot to: {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_filename}: {e}")
        plt.close(fig) # Ensure figure is closed even on save error
        return None, None # Return None if saving failed
    
    plt.show() # Display the plot inline
    plt.close(fig) # Close the figure to free memory
    
    return best_fitness_value, eval_count_at_best # Return results

print("plot_optimization_stage function defined.")


# %% [markdown]
# ## 4. Main Execution Logic

# %% 
def main():
    project_root = find_project_root()
    if project_root is None:
        logging.error("Could not find project root. Place a marker file (.git) or run from within the project.")
        return

    logging.info(f"Project Root: {project_root}")
    results_dir = os.path.join(project_root, RESULTS_DIR_NAME)
    history_filepath = os.path.join(results_dir, HISTORY_FILENAME)
    summary_filepath = os.path.join(results_dir, "final_optimization_fitness.txt") # Define summary file path

    logging.info(f"Looking for history file in: {history_filepath}")

    if not os.path.exists(history_filepath):
        logging.error(f"Optimization history file not found at: {history_filepath}")
        return

    try:
        with open(history_filepath, 'r') as f:
            history_data = json.load(f)
        logging.info(f"Successfully loaded optimization history from: {history_filepath}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {history_filepath}")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading {history_filepath}: {e}")
        return

    # Check if the structure is the new detailed format
    is_detailed_format = isinstance(history_data.get('expert_evolution'), dict) and \
                         isinstance(history_data.get('gating_pso'), dict)

    final_fitness_summary = {} # Store results for final summary

    if not is_detailed_format:
        # --- Handle Simple/Outdated Format ---
        logging.warning("History file format seems outdated or simple (only final values).")
        logging.warning("Attempting to extract final values...")
        # final_fitness_values = {} # Renamed to final_fitness_summary
        for stage, data in history_data.items():
             # Try to extract the best fitness robustly
             fitness = None
             if isinstance(data, list) and data:
                 # Check the last entry first, assuming it might be the best
                 if isinstance(data[-1], dict) and ('best_fitness' in data[-1] or 'gbest' in data[-1]):
                     fitness = data[-1].get('best_fitness', data[-1].get('gbest'))
                 # Fallback to first entry if last didn't work
                 elif isinstance(data[0], dict) and ('best_fitness' in data[0] or 'gbest' in data[0]):
                      fitness = data[0].get('best_fitness', data[0].get('gbest'))
             elif isinstance(data, dict) and ('best_fitness' in data or 'gbest' in data):
                  # Handle case where stage data might be a single dict
                  fitness = data.get('best_fitness', data.get('gbest'))
                  
             if fitness is not None:
                 final_fitness_summary[stage] = {'fitness': fitness, 'evals': 'N/A', 'algorithm': 'N/A'} # Use dict structure
             else:
                 logging.warning(f"Could not parse final fitness for stage '{stage}' from simple format.")
        
        if final_fitness_summary:
            print("\n--- Final Best Fitness Values (from simple format) ---")
            for stage, result in final_fitness_summary.items(): # Iterate through dict
                print(f"  {stage.replace('_', ' ').title()}: {result['fitness']:.6f}")
            print("---------------------------------------------------------")
            # Save this simple summary
            try:
                with open(summary_filepath, 'w') as f:
                    f.write("--- Final Optimization Summary (from simple/outdated format) ---\n")
                    for stage, result in final_fitness_summary.items():
                         f.write(f"Stage: {stage.replace('_', ' ').title()}\n")
                         f.write(f"  Best Fitness: {result['fitness']:.6f}\n\n")
                logging.info(f"Saved simple final fitness values to: {summary_filepath}")
            except Exception as e:
                logging.error(f"Could not save simple final fitness values: {e}")
        else:
            logging.error("Could not extract any meaningful data from the simple history file format.")
        return # Exit after handling simple format

    # --- Process Detailed Format ---    
    logging.info("Detected detailed history format. Generating plots...")
    
    # Ensure the results directory exists for saving plots
    os.makedirs(results_dir, exist_ok=True)
    
    stages_plotted = 0
    
    # Define helper once
    def get_metric_display_name(metric_key):
        name_map = {
            'best_fitness': 'Best Fitness',
            'gbest': 'Global Best Fitness',
            'eval_count': 'Evaluation Count',
            'fevals': 'Function Evaluations',
            'Fevals': 'Function Evaluations',
            'generation': 'Generation',
            'Gen': 'Generation',
            'dx': 'Parameter Convergence (dx)',
            'df': 'Fitness Convergence (df)',
            'Mean Vel.': 'Mean Velocity (PSO)',
            'Mean lbest': 'Mean Local Best (PSO)',
            'Avg. Dist.': 'Average Distance (PSO)',
        }
        return name_map.get(metric_key, metric_key.replace("_", " ").title())
        
    # Get actual eval col names used in the data if available
    expert_hist = history_data.get('expert_evolution', {}).get('history', [])
    gating_hist = history_data.get('gating_pso', {}).get('history', [])
    expert_eval_col_actual = None
    gating_eval_col_actual = None
    if expert_hist:
        expert_eval_col_actual = next((col for col in ['eval_count', 'fevals', 'Fevals', 'generation', 'Gen'] if col in expert_hist[0]), 'Evaluations')
    if gating_hist:
        gating_eval_col_actual = next((col for col in ['eval_count', 'fevals', 'Fevals', 'generation', 'Gen'] if col in gating_hist[0]), 'Evaluations')
        
    # Plot Expert Evolution
    expert_data = history_data.get('expert_evolution')
    if expert_data and isinstance(expert_data, dict):
        algo = expert_data.get('algorithm', 'Unknown_DE') # Provide default algo name
        hist = expert_data.get('history', [])
        best_fitness, eval_count = plot_optimization_stage(hist, 'expert_evolution', algo, results_dir)
        if best_fitness is not None:
             final_fitness_summary['expert_evolution'] = {'fitness': best_fitness, 'evals': eval_count, 'algorithm': algo, 'eval_col_name': expert_eval_col_actual}
             stages_plotted += 1
    else:
        logging.warning("No valid 'expert_evolution' data found in history file.")

    # Plot Gating PSO
    gating_data = history_data.get('gating_pso')
    if gating_data and isinstance(gating_data, dict):
        algo = gating_data.get('algorithm', 'Unknown_PSO') # Provide default algo name
        hist = gating_data.get('history', [])
        best_fitness, eval_count = plot_optimization_stage(hist, 'gating_pso', algo, results_dir)
        if best_fitness is not None:
             final_fitness_summary['gating_pso'] = {'fitness': best_fitness, 'evals': eval_count, 'algorithm': algo, 'eval_col_name': gating_eval_col_actual}
             stages_plotted += 1
    else:
        logging.warning("No valid 'gating_pso' data found in history file.")

    if stages_plotted == 0:
        logging.error("Could not find any valid stage data ('expert_evolution' or 'gating_pso') in the detailed history file.")
        return # Exit if no plots were generated

    # --- Print and Save Final Summary ---
    print("\n--- Final Optimization Summary ---")
    if final_fitness_summary:
        for stage, results in final_fitness_summary.items():
             eval_col_name = results.get('eval_col_name', 'Evaluations') # Get the name used
             eval_label = get_metric_display_name(eval_col_name) # Get the display name
             print(f"  Stage: {stage.replace('_', ' ').title()} ({results.get('algorithm', 'N/A')})")
             print(f"    Best Fitness: {results.get('fitness', 'N/A'):.6f}")
             # Check if eval count is available and not None before printing
             eval_count = results.get('evals')
             if eval_count is not None:
                  print(f"    Achieved at {eval_label}: {eval_count}")
             else:
                  print(f"    {eval_label} not available.")

        print("---------------------------------")

        # Save the detailed summary
        try:
            with open(summary_filepath, 'w') as f:
                f.write("--- Final Optimization Summary ---\n")
                for stage, results in final_fitness_summary.items():
                    eval_col_name = results.get('eval_col_name', 'Evaluations')
                    eval_label = get_metric_display_name(eval_col_name)
                    f.write(f"Stage: {stage.replace('_', ' ').title()} ({results.get('algorithm', 'N/A')})\n")
                    f.write(f"  Best Fitness: {results.get('fitness', 'N/A'):.6f}\n")
                    eval_count = results.get('evals')
                    if eval_count is not None:
                         f.write(f"  Achieved at {eval_label}: {eval_count}\n")
                    else:
                         f.write(f"  {eval_label} not available.\n")
                    f.write("\n") # Add a newline between stages
            logging.info(f"Saved final optimization summary to: {summary_filepath}")
        except Exception as e:
            logging.error(f"Could not save final optimization summary: {e}")
    else:
        print("  No final fitness summary could be generated.")

print("main function defined.")

# %% [markdown]
# ## 5. Run Visualization

# %% 
if __name__ == "__main__":
    print("\n--- Running Optimization Visualization ---")
    main()
    print("--- Visualization Script Finished ---")

# %% [markdown]
# --- End of Script ---

# %%
