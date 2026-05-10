import requests
import json
from pathlib import Path

print("Downloading USPTO-MIT chemical reactions...")
print("(50K real reactions from US Patents with yields)")
print()

# Direct download URLs
urls = {
    "USPTO_50K": "https://raw.githubusercontent.com/wengong-jin/nips2018/master/data/USPTO_50K.txt",
    "USPTO_MIT": "https://raw.githubusercontent.com/wengong-jin/nips2018/master/data/USPTO_MIT.txt"
}

def download_file(name, url):
    try:
        print(f"Downloading {name}...", end=" ", flush=True)
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            output_file = Path(__file__).parent / f"{name}.txt"
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"✓ ({len(lines)} reactions)")
            return output_file, lines
        else:
            print(f"✗ Status {response.status_code}")
            return None, None
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

# Try primary dataset
data_file, reactions = download_file("USPTO_50K", urls["USPTO_50K"])

if not data_file:
    print("\nTrying alternative mirror...")
    data_file, reactions = download_file("USPTO_MIT", urls["USPTO_MIT"])

if data_file:
    print()
    print(f"✓ Successfully downloaded {len(reactions)} reactions")
    print(f"✓ Saved to: {data_file}")
    print()
    
    # Show sample
    print("Sample reaction (SMILES format):")
    print(f"  {reactions[0][:100]}...")
    print()
    
    print("Dataset info:")
    print(f"  Total reactions: {len(reactions)}")
    print(f"  Format: SMILES strings")
    print(f"  Source: US Patents")
    print(f"  Paper: NIPS 2018")
else:
    print("\n✗ Download failed")
    print("\nFallback: Using direct GitHub clone...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/wengong-jin/nips2018.git"],
            cwd=Path(__file__).parent,
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✓ Repository cloned")
        else:
            print(f"✗ Clone failed: {result.stderr.decode()}")
    except Exception as e:
        print(f"✗ Error: {e}")

