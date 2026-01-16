import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from src.utils.logger import g_logger


def plot_metrics(epochs_seen: list[int], examples_seen: list[int], train_values: list[float], val_values: list[float],
                 label: str, savefig_path: str = None, legend_loc: str = None, simplify_x_axis: bool = False) -> None:
    """
    Plot training and validation metrics over epochs and tokens/examples seen.

    Note that the metrics can be for example loss or accuracy, and the 'examples' term here can refer to either tokens
    or examples, depending on the context.

    Args:
        epochs_seen (list[int]): List of epoch numbers.
        examples_seen (list[int]): List of tokens/examples seen.
        train_values (list[float]): List of training metric values.
        val_values (list[float]): List of validation metric values
        label (str): Label for the metric being plotted (e.g., "loss", "accuracy").
        savefig_path (str, optional): Path to save the figure. If None, the plot is shown instead. Defaults to None.
        legend_loc (str, optional): Location of the legend. If None, default location is used. Defaults to None.
        simplify_x_axis (bool, optional): If True, only integer labels are shown on the x-axis. Defaults to False.
    """
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
