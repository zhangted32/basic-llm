#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B-Instruct model from ModelScope.
"""

import os
import sys
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from modelscope.hub.snapshot_download import snapshot_download


def main():
    print("=" * 60)
    print("Downloading Qwen2.5-1.5B-Instruct from ModelScope")
    print("=" * 60)
    print()
    
    # Qwen model on ModelScope
    model_id = "qwen/Qwen2.5-1.5B-Instruct"
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub")
    
    print(f"Model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print()
    print("Starting download... this may take several minutes.")
    print()
    
    try:
        # Download the model from ModelScope
        snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir
        )
        
        # Check if downloaded
        local_dir = os.path.join(cache_dir, "qwen/Qwen2.5-1.5B-Instruct")
        
        print()
        print("✅ Model downloaded successfully from ModelScope!")
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