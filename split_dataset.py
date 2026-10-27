
from pathlib import Path
import random
import shutil
import hashlib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_raw"
DEST = ROOT / "dataset_split"

SEED = 42
random.seed(SEED)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Avoid overwriting an existing split.
if DEST.exists() and any(DEST.rglob("*.*")):
    raise SystemExit(
        "dataset_split already contains files. "
        "Back it up before creating a new split."
    )

for split in ["train", "val", "test"]:
    for label in ["fresh", "spoiled"]:
        (DEST / split / label).mkdir(parents=True, exist_ok=True)

for label in ["fresh", "spoiled"]:
    folder = SOURCE / label

    if not folder.exists():
        raise SystemExit(f"Missing folder: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    # Remove byte-identical duplicates within each class.
    unique = []
    seen = set()

    for p in sorted(files):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(p)

    random.shuffle(unique)

    n = len(unique)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    groups = {
        "train": unique[:n_train],
        "val": unique[n_train:n_train + n_val],
        "test": unique[n_train + n_val:]
    }

    print(f"\n{label.upper()}: {n} unique images")

    for split, images in groups.items():
        for i, src in enumerate(images):
            filename = f"{label}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, DEST / split / label / filename)

        print(f"  {split}: {len(images)}")

print("\nDataset split completed!")
print("Original dataset_raw images are safe.")
