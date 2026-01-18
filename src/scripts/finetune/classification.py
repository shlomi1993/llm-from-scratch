import argparse
import os
import pandas as pd
import time
import torch

from dataclasses import dataclass
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dataset import SpamDataset
from src.model.config import GptConfig
from src.model.gpt import GptModel
from src.scripts.train import format_training_progress
from src.utils.checkpoint import load_model, save_model
from src.utils.device import Device, get_device
from src.utils.logger import g_logger
from src.utils.losses import calc_loss_loader, calc_losses, calc_loss_last_token
from src.utils.tokenization.tokenizer import EOT_IDX, g_tokenizer
from src.utils.visualization import plot_metrics


@dataclass
class ClassificationFineTuningResults:
    """
    Data class to hold the results of classification fine-tuning.
    """
    model: GptModel
    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]
    n_examples_seen: int = None


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a balanced dataset by under-sampling the majority class.

    Args:
        df (pd.DataFrame): Original DataFrame with 'Label' column containing 'spam' and 'ham' entries.

    Returns:
        pd.DataFrame: Balanced dataset with an equal number of 'spam' and 'ham' entries, where 'spam' is labeled as 1
            and 'ham' as 0.
    """

    # Count the instances of "spam"
    n_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(n_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    # Map labels to numerical values
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    return balanced_df


def random_split(df: pd.DataFrame, train_frac: float, validation_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split a DataFrame into training, validation, and test sets.

    Args:
        df (pd.DataFrame): Input DataFrame to be split.
        train_frac (float): Fraction of data to be used for training.
        validation_frac (float): Fraction of data to be used for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing the split DataFrames.
    """

    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def create_classification_dataloaders(training_set_path: str, sep: str, column_names: list[str], train_frac: float,
                                      val_frac: float, save_split_dir: str, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for classification fine-tuning.

    Args:
        training_set_path (str): Path to the training .tsv (tab-separated) file.
        sep (str): Separator used in the .tsv file.
        column_names (list[str]): Column names for the dataset.
        train_frac (float): Fraction of data to be used for training.
        val_frac (float): Fraction of data to be used for validation.
        save_split_dir (str): Directory to save train/val/test splits.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the DataLoaders for training, validation, and test sets.
    """

    # Load and preprocess dataset
    df = pd.read_csv(training_set_path, sep=sep, header=None, names=column_names)
    balanced_df = create_balanced_dataset(df)
    train_df, val_df, test_df = random_split(balanced_df, train_frac, val_frac)  # 70% train, 10% validation, 20% test

    # Save splits to CSV files
    os.makedirs(save_split_dir, exist_ok=True)
    train_csv_path = os.path.join(save_split_dir, "train.csv")
    val_csv_path = os.path.join(save_split_dir, "validation.csv")
    test_csv_path = os.path.join(save_split_dir, "test.csv")
    train_df.to_csv(train_csv_path, index=None)
    val_df.to_csv(val_csv_path, index=None)
    test_df.to_csv(test_csv_path, index=None)

    # Create datasets
    train_dataset = SpamDataset(csv_file=train_csv_path, max_length=None)
    val_dataset = SpamDataset(csv_file=val_csv_path, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(csv_file=test_csv_path, max_length=train_dataset.max_length)

    # Create dataloaders
    torch.manual_seed(seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    return train_loader, val_loader, test_loader


def calc_accuracy_loader(loader: DataLoader, model: GptModel, device: Device, n_batches: int = None) -> float:
    """
    Calculate accuracy of the model on a given DataLoader.

    Args:
        loader (DataLoader): DataLoader to evaluate the model on.
        model (GptModel): The classification model.
        device (Device): Device to perform computation on.
        n_batches (int, optional): Number of batches to evaluate. If None, evaluate on the entire loader. Defaults to None.

    Returns:
        float: Accuracy of the model on the given DataLoader.
    """
    model.eval()
    correct_predictions = 0
    n_examples = 0
    n_batches = len(loader) if n_batches is None else min(n_batches, len(loader))
    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= n_batches:
            break

        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch: Tensor
        target_batch: Tensor
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # Logits of last output token

        predicted_labels = torch.argmax(logits, dim=-1)

        # Update metrics
        n_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    model.train()
    return correct_predictions / n_examples


def load_classifier(model_path: str, device: Device, n_classes: int) -> GptModel:
    """
    Load a fine-tuned classification model from a checkpoint and return it as a GptModel instance in eval mode.

    Args:
        model_path (str): Path to the checkpoint file.
        device (Device): Device to load the model on.
        n_classes (int): Number of output classes.

    Returns:
        GptModel: Loaded classification model.
    """

    # Load the checkpoint to extract config
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = GptConfig(**checkpoint["config"])

    # Create model with the same architecture
    model = GptModel(config)

    # Replace output head with classification head BEFORE loading weights
    model.out_head = torch.nn.Linear(in_features=config.emb_dim, out_features=n_classes)

    # Load the fine-tuned weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    return model


def classify_review(text: str, model: GptModel, device: Device, max_length: int, pad_token_id: int = EOT_IDX) -> tuple[str, float]:
    """
    Classify a single SMS review as "spam" or "not spam" using the fine-tuned classification model.

    Note that this function change the model to eval mode.

    Args:
        text (str): Input text to classify.
        model (GptModel): Fine-tuned classification model.
        device (Device): Device to perform computation on.
        max_length (int): Maximum input length for the model.
        pad_token_id (int, optional): Token ID used for padding. Defaults to PAD_IDX.

    Returns:
        tuple[str, float]: Predicted label ("spam" or "not spam") and confidence score as a float.
    """
    model.eval()

    # Verify that the input length does not exceed model context length
    supported_context = model.pos_emb.weight.shape[0]
    if max_length > supported_context:
        raise ValueError(f"max_length ({max_length}) exceeds model context ({supported_context}).")

    # Tokenize and truncate
    input_ids = g_tokenizer.encode(text)[:max_length]

    # Pad
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits: Tensor = model(input_tensor)[:, -1]

    # Get predicted label and confidence
    probabilities = torch.softmax(logits, dim=-1)
    label_id = torch.argmax(probabilities, dim=-1).item()
    label = "spam" if label_id == 1 else "not spam"
    confidence = probabilities[0, label_id].item()

    # Decode label
    return label, confidence


def finetune_classifier(model: GptModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        device: Device, n_epochs: int, eval_freq, eval_iter) -> ClassificationFineTuningResults:

    """
    Fine-tune a classification model using the provided training and validation DataLoaders.

    Args:
        model (GptModel): The classification model to fine-tune.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        device (Device): Device to perform computation on.
        n_epochs (int): Number of training epochs.
        eval_freq (int): Frequency (in steps) to evaluate model on validation set.
        eval_iter (int): Number of batches to evaluate.

    Returns:
        ClassificationFineTuningResults: Results of the fine-tuning process.
    """

    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # Initialize lists to track losses and accuracies
    example_count = 0
    global_step = -1
    epoch_batches = len(train_loader)
    total_batches = n_epochs * epoch_batches

    try:
        for epoch in range(1, n_epochs + 1):
            for input_batch, target_batch in train_loader:
                input_batch: Tensor

                # Learning step
                model.train()
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_last_token(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients

                # Tracking progress
                example_count += input_batch.shape[0]  # track examples instead of tokens
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = calc_losses(model, train_loader, val_loader, device, eval_iter, calc_loss_last_token)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    progress_msg = format_training_progress(epoch, n_epochs, global_step, total_batches, train_loss, val_loss)
                    g_logger.info(progress_msg)

            # Calculate accuracy after each epoch
            train_accuracy = calc_accuracy_loader(train_loader, model, device, n_batches=eval_iter)
            val_accuracy = calc_accuracy_loader(val_loader, model, device, n_batches=eval_iter)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            progress_msg = format_training_progress(epoch, n_epochs, global_step, total_batches,
                                                    train_acc=train_accuracy * 100, val_acc=val_accuracy * 100)
            g_logger.info(progress_msg)

    except KeyboardInterrupt:
        g_logger.info("Fine-tuning interrupted by user. Returning current model state...")

    return ClassificationFineTuningResults(model, train_losses, val_losses, train_accs, val_accs, example_count)


def run_classification_finetuning_flow(pretrained_model_path: str, tuning_set_path: str, sep="\t",
                                       column_names=["Label", "Text"], train_frac: float = 0.7,
                                       validation_frac: float = 0.1, save_split_dir: str = ".", batch_size: int = 8,
                                       seed: int = 123, device_type: str = "auto", lr: float = 5e-5, n_epochs: int = 5,
                                       weight_decay: float = 0.1, eval_freq: int = 50, eval_iter: int = 5,
                                       loss_plot_save_path: str = None, accuracy_plot_save_path: str = None,
                                       model_save_path: str = "spam-classifier.pth") -> ClassificationFineTuningResults:
    """
    Run the classification fine-tuning flow.

    Args:
        pretrained_model_path (str): Path to a pre-trained foundation GPT2 model.
        tuning_set_path (str): Path to the training .tsv (tab-separated) file.
        sep (str, optional): Separator used in the .tsv file. Defaults to "\t".
        column_names (list[str], optional): Column names for the dataset. Defaults to ["Label", "Text"].
        train_frac (float, optional): Fraction of data to be used for training. Defaults to 0.7.
        validation_frac (float, optional): Fraction of data to be used for validation. Defaults to 0.1.
        save_split_dir (str, optional): Directory to save train/val/test splits. Defaults to ".".
        batch_size (int, optional): Batch size for training. Defaults to 8.
        seed (int, optional): Random seed for reproducibility. Defaults to 123.
        device_type (str, optional): Device to use for training (cpu, cuda, mps, auto). Defaults to "auto".
        lr (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
        n_epochs (int, optional): Number of training epochs. Defaults to 5.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.1.
        eval_freq (int, optional): Evaluation frequency (in steps). Defaults to 50.
        eval_iter (int, optional): Number of batches to evaluate. Defaults to 5.
        loss_plot_save_path (str, optional): Path to save the loss plot. If None, no plot is saved. Defaults to None.
        accuracy_plot_save_path (str, optional): Path to save the accuracy plot. If None, no plot is saved. Defaults to None.
        model_save_path (str, optional): Path to save the fine-tuned model. Defaults to "spam-classifier.pth".

    Returns:
        ClassificationFineTuningResults: Results of the fine-tuning process.
    """
    g_logger.info("Running classification fine-tuning flow...")

    torch.manual_seed(seed)
    device = get_device(device_type)
    g_logger.info(f"Using device '{device.type}' and random seed {seed}")

    g_logger.info("Loading pre-trained model")
    model = load_model(pretrained_model_path, device)[0]
    model.eval()

    g_logger.info("Preparing classification fine-tuning dataset")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        tuning_set_path, sep, column_names, train_frac, validation_frac, save_split_dir, batch_size, seed
    )
    assert train_loader.dataset.max_length <= model.config.context_length, "Dataset sequences are longer than the model's context length."

    g_logger.info("Modifying model for classification fine-tuning")
    for param in model.parameters():
        param.requires_grad = False
    model.out_head = torch.nn.Linear(in_features=model.config.emb_dim, out_features=2)
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    g_logger.info("Moving model to device")
    model.to(device)

    g_logger.info(f"Setting up optimizer with learning rate of {lr} and weight decay of {weight_decay}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    torch.manual_seed(seed)
    g_logger.info("Starting classification fine-tuning...")
    start_time = time.time()
    results = finetune_classifier(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    g_logger.info(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    g_logger.info("Evaluation on fine-tuned model:")
    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, calc_loss_last_token, model, device, n_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, calc_loss_last_token, model, device, n_batches=eval_iter)
        test_loss = calc_loss_loader(test_loader, calc_loss_last_token, model, device, n_batches=eval_iter)
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print()
    g_logger.info(f"  Training:   loss = {train_loss:.3f}, accuracy = {train_accuracy * 100:.2f}%")
    g_logger.info(f"  Validation: loss = {val_loss:.3f}, accuracy = {val_accuracy * 100:.2f}%")
    g_logger.info(f"  Test:       loss = {test_loss:.3f}, accuracy = {test_accuracy * 100:.2f}%")

    if loss_plot_save_path:
        loss_epochs_tensor = torch.linspace(0, n_epochs, len(results.train_losses))
        loss_examples_seen_tensor = torch.linspace(0, results.n_examples_seen, len(results.train_losses))
        plot_metrics(loss_epochs_tensor, loss_examples_seen_tensor, results.train_losses, results.val_losses,
                     savefig_path=loss_plot_save_path, label="loss", legend_loc="upper right")

    if accuracy_plot_save_path:
        accu_epochs_tensor = torch.linspace(0, n_epochs, len(results.train_accuracies))
        accu_examples_seen_tensor = torch.linspace(0, results.n_examples_seen, len(results.train_accuracies))
        plot_metrics(accu_epochs_tensor, accu_examples_seen_tensor, results.train_accuracies, results.val_accuracies,
                     savefig_path=accuracy_plot_save_path, label="accuracy")

    save_model(model, model_save_path, optimizer)

    return results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add command-line arguments to the parser for classification fine-tuning.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """
    parser.add_argument("--pretrained-model-path", type=str, required=True, help="Path to a pre-trained foundation GPT2 model.")
    parser.add_argument("--tuning-set-path", type=str, required=True, help="Path to the training .tsv (tab-separated) file.")
    parser.add_argument("--column-names", type=str, nargs="+", default=["Label", "Text"], help="Column names for the dataset.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of data for training.")
    parser.add_argument("--validation-frac", type=float, default=0.1, help="Fraction of data for validation.")
    parser.add_argument("--save-split-dir", type=str, default=".", help="Directory to save train/val/test splits.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, cuda, mps, auto).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--n-epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency (in steps).")
    parser.add_argument("--eval-iter", type=int, default=5, help="Number of batches to evaluate.")
    parser.add_argument("--loss-plot-save-path", type=str, default=None, help="Path to save loss plot (None to skip).")
    parser.add_argument("--accuracy-plot-save-path", type=str, default=None, help="Path to save accuracy plot (None to skip).")
    parser.add_argument("--model-save-path", type=str, default="spam_classifier.pth", help="Path to save the fine-tuned model.")


def main() -> None:
    """
    Main function to run the classification fine-tuning flow. Called when the script is executed directly.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT model for SMS spam classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    run_classification_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        tuning_set_path=args.tuning_set_path,
        column_names=args.column_names,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        save_split_dir=args.save_split_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        accuracy_plot_save_path=args.accuracy_plot_save_path,
        model_save_path=args.model_save_path
    )


if __name__ == "__main__":
    main()
