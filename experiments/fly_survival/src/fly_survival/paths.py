from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "outputs/fly_survival"
PREVIEWS = ROOT / "previews/fly_survival"
NEURAL_DATA = OUTPUTS / "malecns"
VENDOR = ROOT / "experiments/fly_survival/third_party/fly_ai"
