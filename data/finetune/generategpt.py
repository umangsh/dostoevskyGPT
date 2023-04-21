"""Generate tokens from fine-tuned DostoevskyGPT."""
import argparse
import time

import tiktoken
import torch

from common import nanogpt


def generate_samples(  # pylint: disable=too-many-arguments, too-many-locals
    checkpoint_file: str,
    device: str,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> None:
    """Generate text sample from fine-tuned GPT model based on a prompt."""
    time_seed = int(time.time())
    torch.manual_seed(time_seed)
    ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

    # init from a model saved in a specific directory
    checkpoint = torch.load(checkpoint_file, map_location=device)
    gptconf = nanogpt.GPTConfig(**checkpoint["model_args"])
    model = nanogpt.GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, _v in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    prompt_tensor = torch.tensor(
        enc.encode(prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device
    )[None, ...]

    # run generation
    with torch.no_grad(), ctx:
        for _unused in range(num_samples):
            sample = model.generate(prompt_tensor, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(sample[0].tolist()))  # noqa: T201
            print("-" * 80)  # noqa: T201


def main() -> None:
    """Main method."""
    # Initialize sample parameters
    checkpoint_file = "/Users/umang/Desktop/github/dostoevskyGPT/out/finetune/checkpoint.pt"
    device = "mps"
    num_samples = 10  # number of samples to draw
    max_new_tokens = 500  # number of tokens generated in each sample
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability

    parser = argparse.ArgumentParser(description="Generate text from fine-tuned DostoevskyGPT.")
    parser.add_argument("-s", "--start", type=str, help="start prompt")
    args = parser.parse_args()

    generate_samples(checkpoint_file, device, args.start, num_samples, max_new_tokens, temperature, top_k)


if __name__ == "__main__":
    main()
