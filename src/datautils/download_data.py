import json
from datasets import load_dataset

# This is test code to download the data local and run initial setup for bench marking.

print("Streaming a small slice of FineWeb-Edu...")
# By using streaming=True, we avoid downloading the 2.15GB file shard!
stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

sample_data = []
for idx, example in enumerate(stream):
    sample_data.append({"text": example["text"]})
    if idx >= 4999: # Grab exactly 5000 clean text documents
        break

# Write out to a small local file (~15-20MB total)
with open("./fineweb_sample_5k.jsonl", "w", encoding="utf-8") as f:
    for item in sample_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Sample file generated successfully: ./fineweb_sample_5k.jsonl")
