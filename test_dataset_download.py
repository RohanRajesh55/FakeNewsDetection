import kagglehub
import os
from pathlib import Path

print("Starting dataset download test...")

try:
    # Download dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    
    # Verify download path
    print(f"\nDataset downloaded to: {path}")
    
    # Check if files exist
    if Path(path).exists():
        print("\nVerification successful! Files found at:")
        for root, dirs, files in os.walk(path):
            for file in files:
                print(f"- {os.path.join(root, file)}")
    else:
        print("\nError: Download directory not found")
        
except Exception as e:
    print(f"\nError during download: {str(e)}")
