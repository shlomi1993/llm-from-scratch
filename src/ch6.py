import pandas as pd
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


DATA_FILE_PATH = "/Users/shlomibenshushan/Repositories/llm-from-scratch/datasets/sms_spam_collection/SMSSpamCollection.tsv"


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    # Map labels to numerical values
    balanced_df["Label"].map({"ham": 0, "spam": 1})

    return balanced_df


def random_split(df: pd.DataFrame, train_frac: float, validation_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


class SpamDataset(Dataset):

    def __init__(self, csv_file: str, tokenizer: tiktoken.Encoding, max_length: int = None, pad_token_id: int = 50256) -> None:
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Set max length: either user-provided or longest sample
        self.max_length = max_length or max(len(et) for et in self.encoded_texts)

        # Truncate sequences if they are longer than max_length
        self.encoded_texts = [et[:self.max_length] for et in self.encoded_texts]

        # Pad sequences to the longest sequence
        self.encoded_texts = [et + [pad_token_id] * (self.max_length - len(et)) for et in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        encoded = torch.tensor(self.encoded_texts[index], dtype=torch.long)
        label = torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long)
        return encoded, label


# Flow
df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 70% train, 10% validation, 20% test
# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/train.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="datasets/sms_spam_collection/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
# Test: Verify that train_dataset.max_length == val_dataset.max_length == test_dataset.max_length == 120
torch.manual_seed(123)
num_workers = 0
batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
# Test: For each loader, the input batch shape is [8, 120], and the label batch shape is [8]
# Test: Verify that len(train_loader) == 130, len(val_loader) == 19, len(test_loader) == 38

# TODO Stopped at 6.4