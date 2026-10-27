from pathlib import Path
from urllib.request import urlretrieve
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

url = "https://github.com/Suf-tech/BreadGuard-AI/releases/download/v1.0.0/breadguard_trained_models.zip"
zip_path = ROOT / "breadguard_models_download.zip"

print("Downloading trained models...")
urlretrieve(url, zip_path)

print("Extracting models...")
with zipfile.ZipFile(zip_path) as archive:
    for member in archive.infolist():
        if member.is_dir() or not member.filename.lower().endswith(".keras"):
            continue
        destination = MODELS / Path(member.filename).name
        with archive.open(member) as source, destination.open("wb") as target:
            import shutil
            shutil.copyfileobj(source, target)
        print("Extracted:", destination.name)

zip_path.unlink()

required = ["best_mobilenetv2.keras", "bread_scratch_cnn.keras"]
missing = [name for name in required if not (MODELS / name).exists()]
if missing:
    raise RuntimeError(f"Required models missing from ZIP: {missing}")

print("Both trained models are ready!")
