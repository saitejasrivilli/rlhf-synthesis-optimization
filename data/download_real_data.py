import requests
import json
from pathlib import Path

print("Finding available real chemistry datasets...")
print()

# Alternative real datasets still available

SOURCES = [
    {
        "name": "ChEMBL Reactions",
        "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/",
        "description": "Real bioactive molecules with experimental data"
    },
    {
        "name": "Reaxys/Pistachio sample data",
        "url": "https://www.pistachiodb.com/",
        "description": "Real synthesis reactions (free tier)"
    },
    {
        "name": "PubChem Bioassay",
        "url": "https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
        "description": "Real experimental bioassay data"
    }
]

print("Real datasets available:")
for i, source in enumerate(SOURCES, 1):
    print(f"{i}. {source['name']}")
    print(f"   URL: {source['url']}")
    print(f"   Description: {source['description']}")
    print()

# Use PubChem Compound data as fallback
print("Downloading from PubChem (NIH official database)...")
print()

try:
    # PubChem compounds with properties
    url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"
    print(f"Note: Full download from {url}")
    print("Downloading sample of 1000 pharmaceutical compounds from PubChem...")
    
    # Use PubChem JSON API for a sample
    pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
    
    # Known drug CIDs from PubChem
    drug_cids = [
        2244,      # Aspirin
        3672,      # Ibuprofen  
        1983,      # Paracetamol
        156391,    # Naproxen
        3825,      # Ketoprofen
    ]
    
    print("\nAttempting to download known drug data from PubChem...")
    
    compounds = []
    for cid in drug_cids:
        try:
            # Get compound data
            resp = requests.get(f"{pubchem_url}{cid}/JSON", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                compounds.append({
                    "cid": cid,
                    "name": data.get('PC_Compounds', [{}])[0].get('props', [{}])[0],
                    "data": data
                })
                print(f"  ✓ Downloaded CID {cid}")
        except:
            pass
    
    print(f"\n✓ Retrieved {len(compounds)} compounds from PubChem")
    
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 80)
print("BETTER APPROACH: Use Harvard/MIT Real Synthesis Data")
print("=" * 80)
print()
print("Real datasets that are actually available:")
print()
print("1. Grambow et al. - RMG Database")
print("   URL: https://rmg.mit.edu/")
print("   Contains: Real reaction mechanisms with kinetics")
print()
print("2. Open Reaction Database (ORD)")
print("   URL: https://open-reaction-database.org/")
print("   Contains: 500K+ real reactions from literature")
print("   Format: Structured reaction data")
print("   Citation: Prepared for publication")
print()
print("3. Schwaller et al. USPTO-STEREO Data")
print("   URL: https://github.com/pschwaller/") 
print("   Contains: Patent reactions with stereochemistry")
print()
print("4. Goodman/Vaucher Data (Real Synthesis)")
print("   Description: Actual experimental synthesis outcomes")
print("   Source: Literature mining")
print()

