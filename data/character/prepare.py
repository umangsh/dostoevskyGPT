"""Prepare training and validation data splits for training a character based model
with Dostoevsky's works."""
import pickle

import numpy


def encode(string: str, int_map: dict) -> list[int]:
    """Encode a string into a list of integers."""
    return [int_map[c] for c in string]


def prepare_dataset(  # pylint: disable=too-many-locals
    input_file: str, train_file: str, validation_file: str, meta_file: str
) -> None:
    """Process the dataset available at input_file path."""
    with open(input_file, encoding="utf-8") as file:
        content = file.read()
    print(f"content has {len(content):,} characters")  # noqa: T201

    # get all the unique characters that occur in this text
    chars = sorted(set(content))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))  # noqa: T201
    print(f"vocab size: {vocab_size:,}")  # noqa: T201

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = dict(enumerate(chars))

    # create the train and test splits
    train_data = content[: int(len(content) * 0.9)]
    train_ids = encode(train_data, stoi)
    numpy.array(train_ids, dtype=numpy.uint16).tofile(train_file)

    val_data = content[int(len(content) * 0.9) :]
    val_ids = encode(val_data, stoi)
    numpy.array(val_ids, dtype=numpy.uint16).tofile(validation_file)

    print(f"train has {len(train_ids):,} tokens")  # noqa: T201
    print(f"val has {len(val_ids):,} tokens")  # noqa: T201

    # save the meta information, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(meta_file, "wb") as file:
        pickle.dump(meta, file)


def main() -> None:
    """Main method."""
    input_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/dataset.txt"
    train_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/character/train.bin"
    validation_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/character/val.bin"
    meta_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/character/meta.pkl"

    # Generate the training and validation datasets from the input file.
    prepare_dataset(input_file, train_file, validation_file, meta_file)


if __name__ == "__main__":
    main()
