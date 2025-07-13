"""
Dataset preparation script, that loads the chunked function dataset 
and saves it as a Hugging Face dataset for fine-tuning.
"""

from datasets import Dataset

# Load chunked JSONL (from clean_code.py output)
dataset = Dataset.from_json("data/chunks/github_code_chunks.jsonl")

# Save to disk for use in LoRA fine-tuning
dataset.save_to_disk("datasets/GitLLM_data")

print(f"✅ Saved dataset with {len(dataset)} samples to datasets/GitLLM_data")
