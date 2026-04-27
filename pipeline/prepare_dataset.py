"""
Dataset preparation script, that loads the chunked function dataset 
and saves it as a Hugging Face dataset for fine-tuning.
"""
from pathlib import Path
from datasets import Dataset

SAVE_PATH = "datasets/GitLLM_data"

def main():
    chunk_path = Path("data/chunks/github_code_chunks.jsonl")

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_path}"
        )
    # Load chunked JSONL (from clean_code.py output)
    dataset = Dataset.from_json(
        chunk_path
    )

    # Save to disk for use in LoRA fine-tuning
    dataset.save_to_disk(
        SAVE_PATH
    )

    print(
        f"✅ Saved dataset with {len(dataset)} samples "
        f"to {SAVE_PATH}"
    )


if __name__ == "__main__":
    main()
