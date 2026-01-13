import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from src.utils.logger import g_logger


def plot_metrics(epochs_seen: list[int], examples_seen: list[int], train_values: list[float], val_values: list[float],
                 label: str, savefig_path: str = None, legend_loc: str = None, simplify_x_axis: bool = False) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.title())
    ax1.legend(loc=legend_loc)
    if simplify_x_axis:
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens/examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens/Examples Seen")

    fig.tight_layout()  # Adjust layout to make room
    if savefig_path:
        parent_dir = os.path.dirname(savefig_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(savefig_path)
        g_logger.info(f"Saved {label} plot to {savefig_path}")
    else:
        plt.show()
