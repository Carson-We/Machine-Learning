#!/usr/bin/env python3
"""
LEPAUTE Dataset → WebDataset Converter (Resumable & Production-Ready)
"""

import json
import os
import tarfile
from pathlib import Path
import webdataset as wds
from tqdm import tqdm


class ResumePattern:
    """
    A proxy string class to trick WebDataset's ShardWriter into starting
    from a specific shard index without overwriting existing files.
    """
    def __init__(self, pattern: str, start_shard: int):
        self.pattern = pattern
        self.start_shard = start_shard

    def __mod__(self, shard_index: int):
        # ShardWriter generates indices starting from 0.
        # We offset it by the starting shard derived from existing files.
        return self.pattern % (shard_index + self.start_shard)


def find_json_files(split_dir: Path):
    """Finds all .json files in the split directory."""
    return sorted(split_dir.glob("*.json"))


def load_record(json_path: Path) -> dict:
    """Safely loads a single JSON record."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resumption_state(output_dir: Path, split_name: str):
    """
    Analyzes existing .tar files to determine where to resume.
    Returns the start_shard index and a set of already processed dataset keys.
    """
    existing_tars = sorted(output_dir.glob(f"{split_name}-*.tar"))
    
    if not existing_tars:
        return 0, set()
        
    # Extract shard indices
    shard_indices = []
    for tar in existing_tars:
        try:
            # Format expected: split_name-000123.tar
            idx = int(tar.stem.split('-')[-1])
            shard_indices.append((idx, tar))
        except ValueError:
            continue
            
    if not shard_indices:
        return 0, set()
        
    shard_indices.sort(key=lambda x: x[0])
    max_idx, max_tar = shard_indices[-1]
    
    # ROLLBACK: Delete the highest shard. 
    # If the script crashed (e.g., No space left on device), the last tar is guaranteed
    # to be incomplete. Rolling back one shard ensures absolute data integrity.
    print(f"Rolling back incomplete/latest shard to ensure integrity: {max_tar.name}")
    max_tar.unlink()
    
    # Retain the rest as valid shards
    valid_tars = [tar for idx, tar in shard_indices[:-1]]
    
    if not valid_tars:
        return 0, set()
        
    processed_keys = set()
    print(f"Scanning {len(valid_tars)} intact shards to build resumption cache (this is fast)...")
    for tar_path in tqdm(valid_tars, desc="Scanning Shards"):
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Scanning tar headers is extremely fast compared to reading contents
                for name in tar.getnames():
                    if name.endswith('.json'):
                        processed_keys.add(Path(name).stem)
        except tarfile.ReadError:
            print(f"Error reading {tar_path}. The archive might be heavily corrupted. Delete it and restart.")
            raise
            
    # The new starting shard index will be the next sequential number
    start_shard = shard_indices[-2][0] + 1 if len(shard_indices) > 1 else 0
    return start_shard, processed_keys


def convert_split(split_dir: Path, output_dir: Path, split_name: str, 
                  maxcount: int = 8000, maxsize: int = 800_000_000):
    """
    Converts a single split (train or test) with full resumption support.
    """
    json_files = find_json_files(split_dir)
    if not json_files:
        print(f"No .json files found in {split_name} directory.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Evaluate Resumption State
    start_shard, processed_keys = get_resumption_state(output_dir, split_name)
    
    if processed_keys:
        original_count = len(json_files)
        # Filter out records that are securely packed in existing shards
        json_files = [f for f in json_files if f.stem not in processed_keys]
        print(f"Resuming: Skipped {original_count - len(json_files)} safely processed records.")
        
    if not json_files:
        print(f"All records for {split_name} are already processed!")
        return

    print(f"Processing {len(json_files)} remaining records for {split_name} split...")

    shard_pattern = str(output_dir / f"{split_name}-%06d.tar")
    # 2. Inject the proxy pattern to offset the starting index
    resume_pattern = ResumePattern(shard_pattern, start_shard)

    with wds.ShardWriter(resume_pattern, maxcount=maxcount, maxsize=maxsize) as writer:
        for json_path in tqdm(json_files, desc=f"Converting {split_name}"):
            try:
                record = load_record(json_path)
                
                # Derive key safely
                key = Path(record.get("frame_a", json_path.stem)).stem
                
                frame_a_path = split_dir / record.get("frame_a", "")
                frame_b_path = split_dir / record.get("frame_b", "")
                
                # 3. Data Integrity Validation: Skip missing files to prevent dataset poison
                if not frame_a_path.is_file() or not frame_b_path.is_file():
                    print(f"Missing images for record {key}. Skipping to maintain dataset integrity.")
                    continue
                    
                sample = {"__key__": key}
                sample["json"] = json.dumps(record, ensure_ascii=False).encode("utf-8")
                
                # 4. Safe Resource Management
                with open(frame_a_path, "rb") as fa:
                    sample["frame_a.jpg"] = fa.read()
                    
                with open(frame_b_path, "rb") as fb:
                    sample["frame_b.jpg"] = fb.read()
                
                writer.write(sample)
                
            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")
                continue

    print(f"{split_name} split conversion complete!")


def main():
    # ================== Parameters ==================
    input_root = Path("./dataset/lepaute_dataset")          
    output_root = Path("./dataset/lepaute_webdataset")     
    
    maxcount = 8000          # Max records per tar
    maxsize = 900_000_000    # Max bytes per tar (~900MB)
    # ================================================
    
    print("Starting LEPAUTE to WebDataset conversion...")
    
    for split in ["train", "test"]:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f"Cannot find {split} directory, skipping.")
            continue
        
        output_dir = output_root / split
        convert_split(split_dir, output_dir, split, maxcount, maxsize)
    
    print("\nAll splits successfully converted!")
    print(f"Output location: {output_root.resolve()}")
    print("\nNext step: Use upload-large-folder to upload the shards.")


if __name__ == "__main__":
    main()