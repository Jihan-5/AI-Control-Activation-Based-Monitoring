"""Print the current commit SHA of a Hugging Face model repo (pin it before any reported run).

Usage: python scripts/pin_revision.py Qwen/Qwen2.5-1.5B-Instruct
"""

import sys

from huggingface_hub import model_info

if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    print(model_info(repo).sha)
