from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def get_data_path(name=""):
    return PROJECT_ROOT / "data" / name if name else PROJECT_ROOT / "data"

def get_models_dir(name=""):
    path = PROJECT_ROOT / "models" / "checkpoints" / name if name else PROJECT_ROOT / "models" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_results_dir():
    path = PROJECT_ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path

def initialize_directories():
    for d in [get_data_path("raw"), get_data_path("processed"), get_data_path("labeled"), 
              get_results_dir()]:
        d.mkdir(parents=True, exist_ok=True)
