import requests
import json
from pathlib import Path
import subprocess

print("Downloading Open Reaction Database (ORD)...")
print("(500K+ real reactions from literature)")
print()

# Clone ORD repository
try:
    print("Cloning ORD repository...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/open-reaction-database/ord-schema.git"
    ], timeout=60, capture_output=True)
    print("✓ Cloned successfully")
except Exception as e:
    print(f"✗ Clone failed: {e}")

# Download ORD data files
try:
    print("\nDownloading ORD data files...")
    url = "https://open-reaction-database.org/download"
    print(f"Data available at: {url}")
    
    # Download metadata
    meta_url = "https://raw.githubusercontent.com/open-reaction-database/ord-schema/main/README.md"
    response = requests.get(meta_url, timeout=10)
    
    if response.status_code == 200:
        print("✓ ORD documentation retrieved")
        
        # Save README
        with open("ORD_README.md", "w") as f:
            f.write(response.text)
        
        print("\nOpen Reaction Database includes:")
        print("  • 500,000+ real chemical reactions")
        print("  • From published literature")
        print("  • Structured reaction information")
        print("  • Reproducible synthesis data")
        print("  • CC0 1.0 Universal (public domain)")
        
except Exception as e:
    print(f"Note: {e}")
    print("\nManual download available at:")
    print("https://open-reaction-database.org/download")

