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
    """
    if not history_data:
        logging.warning(f"No history data found for stage '{stage_name}'. Skipping plot.")
        return

    df = pd.DataFrame(history_data)

    # Check for required columns used for plotting
    required_cols = ['eval_count', 'best_fitness']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.warning(f"History data for '{stage_name}' is missing required columns {missing}. Skipping plot.")
        return

    # Determine metrics to plot besides best_fitness
    metrics_to_plot = [col for col in df.columns if col not in ['eval_count', 'best_fitness']]

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(12, 6 * (num_metrics + 1)), sharex=True)
    
    # Ensure axes is always iterable, even if only one plot
    if num_metrics == 0:
        axes = [axes] # Make it a list containing the single Axes object
    else:
        axes = axes.flatten()

    sns.set_style("whitegrid")

    # Plot primary fitness (using best_fitness)
    sns.lineplot(ax=axes[0], data=df, x='eval_count', y='best_fitness', marker='o', label='Best Fitness', color='#1f77b4')
    axes[0].set_title(f'{stage_name.replace("_", " ").title()} Convergence (Algorithm: {algorithm.upper()})', fontsize=14)
    axes[0].set_ylabel('Best Fitness (Lower is Better)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot auxiliary metrics
    color_palette = sns.color_palette("viridis", num_metrics)
    for i, metric in enumerate(metrics_to_plot):
        ax_idx = i + 1
        sns.lineplot(ax=axes[ax_idx], data=df, x='eval_count', y=metric, marker='.', label=metric.replace("_", " ").title(), color=color_palette[i])
        axes[ax_idx].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Evaluation Count', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    plot_filename = os.path.join(output_dir, f'{stage_name}_convergence.png')
    
    try:
        plt.savefig(plot_filename)
        logging.info(f"Saved {stage_name} convergence plot to: {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_filename}: {e}")
    
    plt.show() # Display the plot inline
    
    plt.close(fig) # Close the figure to free memory

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

    if not is_detailed_format:
        logging.warning("History file format seems outdated or simple (only final values).")
        logging.warning("Attempting to extract final values...")
        final_fitness_values = {}
        for stage, data in history_data.items():
            if isinstance(data, list) and data and isinstance(data[0], dict) and 'best_fitness' in data[0]:
                final_fitness_values[stage] = data[0]['best_fitness']
            else:
                 logging.warning(f"Could not parse final fitness for stage '{stage}'")
        
        if final_fitness_values:
            print("\n--- Final Best Fitness Values (from simple format) ---")
            for stage, fitness in final_fitness_values.items():
                print(f"  {stage.replace('_', ' ').title()}: {fitness:.6f}")
            print("---------------------------------------------------------")
            # Optionally save these simple values too
            simple_output_path = os.path.join(results_dir, 'final_optimization_fitness_simple.txt')
            try:
                with open(simple_output_path, 'w') as f:
                    f.write("--- Final Best Fitness Values (from simple format) ---\n")
                    for stage, fitness in final_fitness_values.items():
                         f.write(f"  {stage.replace('_', ' ').title()}: {fitness:.6f}\n")
                logging.info(f"Saved simple final fitness values to: {simple_output_path}")
            except Exception as e:
                logging.error(f"Could not save simple final fitness values: {e}")
        else:
            logging.error("Could not extract any meaningful data from the history file.")
        return # Exit after handling simple format

    # --- Process Detailed Format ---    
    logging.info("Detected detailed history format. Generating plots...")
    
    # Ensure the results directory exists for saving plots
    os.makedirs(results_dir, exist_ok=True)
    
    stages_plotted = 0
    # Plot Expert Evolution
    expert_data = history_data.get('expert_evolution')
    if expert_data and isinstance(expert_data, dict):
        algo = expert_data.get('algorithm', 'Unknown')
        hist = expert_data.get('history', [])
        plot_optimization_stage(hist, 'expert_evolution', algo, results_dir)
        stages_plotted += 1
    else:
        logging.warning("No valid 'expert_evolution' data found in history file.")

    # Plot Gating PSO
    gating_data = history_data.get('gating_pso')
    if gating_data and isinstance(gating_data, dict):
        algo = gating_data.get('algorithm', 'Unknown')
        hist = gating_data.get('history', [])
        plot_optimization_stage(hist, 'gating_pso', algo, results_dir)
        stages_plotted += 1
    else:
        logging.warning("No valid 'gating_pso' data found in history file.")

    if stages_plotted == 0:
        logging.error("Could not find any valid stage data ('expert_evolution' or 'gating_pso') in the detailed history file.")

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
