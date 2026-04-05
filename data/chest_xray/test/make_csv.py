from pathlib import Path
import csv

base = Path.cwd().parent.parent.parent
output = Path("data/val_labels.csv")

rows = []

for class_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
    class_dir = base / "data" / "chest_xray" / "test" / class_name
    if not class_dir.exists():
        print(f"Warning: folder not found: {class_dir}")
        continue

    for img_path in sorted(class_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            rows.append([str(img_path), label])

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print(f"Created {output} with {len(rows)} rows.")
