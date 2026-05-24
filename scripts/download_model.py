#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B-Instruct model using huggingface-hub Python API.
"""

import os
import sys
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Downloading Qwen2.5-1.5B-Instruct Model")
    print("=" * 60)
    print()
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    
    print(f"Model: {model_id}")
    print(f"Download location: {local_dir}")
    print()
    print("Starting download... this may take several minutes.")
    print()
    
    try:
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print()
        print("✅ Model downloaded successfully!")
        print(f"   Location: {local_dir}")
        
        # Verify download
        if os.path.exists(local_dir):
            files = os.listdir(local_dir)
            print(f"   Files downloaded: {len(files)}")
            for f in files[:5]:  # Show first 5 files
                print(f"     - {f}")
            if len(files) > 5:
                print(f"     ... and {len(files) - 5} more files")
                
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print("Please check your network connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()