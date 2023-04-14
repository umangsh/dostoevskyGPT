"""Prepare training and validation data splits for finetuning GPT-2 with
Dostoevsky's works."""
import numpy as np
import tiktoken


def prepare_dataset(input_file: str, train_file: str, validation_file: str) -> None:
    """Process the dataset available at input_file path."""
    with open(input_file, encoding="utf-8") as file:
        data = file.read()

    train_data = data[: int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9) :]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")  # noqa: T201
    print(f"val has {len(val_ids):,} tokens")  # noqa: T201

    # export to files
    np.array(train_ids, dtype=np.uint16).tofile(train_file)
    np.array(val_ids, dtype=np.uint16).tofile(validation_file)


def main():
    """Main method."""
    input_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/dataset.txt"
    train_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/finetune/train.bin"
    validation_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/finetune/val.bin"

    # Generate the training and validation datasets from the input file.
    prepare_dataset(input_file, train_file, validation_file)


if __name__ == "__main__":
    main()
