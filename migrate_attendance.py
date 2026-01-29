import csv
import os
from datetime import datetime


OLD_FILE = "attendance.csv"
NEW_FILE = "attendance_v2.csv"


def _is_v1(path: str) -> bool:
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        return [h.strip().lower() for h in header] == ["name", "datetime"]
    except Exception:
        return False


def migrate(old_path: str = OLD_FILE, new_path: str = NEW_FILE) -> None:
    """
    Convert old v1 attendance.csv (name,datetime ISO) into v2 format:
      name,date,punch_in,punch_out

    Rules:
    - Each v1 record becomes a v2 row with punch_in set to the time and punch_out empty.
    - If v2 already has a row for same (name,date), do not overwrite.
    """
    if not os.path.isfile(old_path):
        raise FileNotFoundError(f"{old_path} not found")
    if not _is_v1(old_path):
        raise RuntimeError(f"{old_path} is not v1 format (name,datetime)")

    existing = {}
    if os.path.isfile(new_path):
        with open(new_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("name"), row.get("date"))
                existing[key] = row

    to_add = []
    with open(old_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            dt_str = (row.get("datetime") or "").strip()
            if not name or not dt_str:
                continue
            try:
                dt = datetime.fromisoformat(dt_str)
            except Exception:
                continue

            date = dt.date().isoformat()
            punch_in = dt.strftime("%H:%M:%S")
            key = (name, date)
            if key in existing:
                continue

            to_add.append({"name": name, "date": date, "punch_in": punch_in, "punch_out": ""})
            existing[key] = to_add[-1]

    # Write v2 file
    with open(new_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "date", "punch_in", "punch_out"])
        writer.writeheader()
        # stable ordering
        for key in sorted(existing.keys(), key=lambda k: (k[0] or "", k[1] or "")):
            writer.writerow(existing[key])

    print(f"Migrated {len(to_add)} records from {old_path} -> {new_path}")


if __name__ == "__main__":
    migrate()


