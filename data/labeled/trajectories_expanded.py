import json
import random
from pathlib import Path

# Base trajectories
base_trajectories = [
    {
        "molecule": "Aspirin",
        "trajectory": [
            "Dissolve salicylic acid in acetic anhydride",
            "Heat to 80°C for 2 hours with stirring",
            "Cool slowly to room temperature",
            "Quench with water dropwise",
            "Filter and recrystallize"
        ],
        "yield": 0.95,
        "selectivity": 0.98,
        "safety_risk": 0.1
    },
    {
        "molecule": "Ibuprofen",
        "trajectory": [
            "Friedel-Crafts acylation at 0°C",
            "Add AlCl3 catalyst",
            "Warm to 100°C over 4 hours",
            "Quench with HCl",
            "Extract with ethyl acetate",
            "Recrystallize"
        ],
        "yield": 0.88,
        "selectivity": 0.92,
        "safety_risk": 0.2
    },
    {
        "molecule": "Paracetamol",
        "trajectory": [
            "Reduce p-nitrophenol with iron/AcOH",
            "Heat to 60°C",
            "Filter product",
            "Acetylate",
            "Recrystallize",
            "Dry"
        ],
        "yield": 0.92,
        "selectivity": 0.95,
        "safety_risk": 0.15
    },
    {
        "molecule": "Naproxen",
        "trajectory": [
            "Friedel-Crafts acylation",
            "Install protecting group",
            "Reduce ketone",
            "Oxidative coupling",
            "Install carboxylic acid",
            "Resolve enantiomers",
            "Crystallize"
        ],
        "yield": 0.75,
        "selectivity": 0.88,
        "safety_risk": 0.3
    },
    {
        "molecule": "Ketoprofen",
        "trajectory": [
            "Friedel-Crafts acylation",
            "Install benzoyl group",
            "Second acylation",
            "Install COOH",
            "Resolve enantiomers",
            "Crystallize"
        ],
        "yield": 0.82,
        "selectivity": 0.90,
        "safety_risk": 0.25
    }
]

# Generate 500 variations
all_trajectories = []
for traj in base_trajectories:
    for i in range(100):  # 100 variations per molecule = 500 total
        var = traj.copy()
        # Add realistic noise
        var["yield"] = max(0.5, min(1.0, var["yield"] + random.gauss(0, 0.04)))
        var["selectivity"] = max(0.5, min(1.0, var["selectivity"] + random.gauss(0, 0.03)))
        var["temperature"] = random.randint(40, 140)
        var["time_hours"] = random.randint(1, 10)
        all_trajectories.append(var)

Path("../").mkdir(parents=True, exist_ok=True)
with open("../trajectories_expanded.jsonl", 'w') as f:
    for traj in all_trajectories:
        f.write(json.dumps(traj) + '\n')

print(f"✓ Generated {len(all_trajectories)} trajectories")
