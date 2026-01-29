import os
import csv
import logging
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "attendance.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


ATTENDANCE_FILE = "attendance.csv"
ATTENDANCE_V1_BACKUP = "attendance_v1_backup.csv"


def ensure_attendance_file(path: str = ATTENDANCE_FILE) -> str:
    """
    Ensure the attendance CSV exists with header.

    New format (v2):
      name,date,punch_in,punch_out

    Backward compatibility:
      If an old v1 file exists with columns [name, datetime], we keep it as-is
      and will append using v2 by creating a new file `attendance_v2.csv`.
    """
    file_exists = os.path.isfile(path)

    if not file_exists:
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "date", "punch_in", "punch_out"])
            logging.info("Created new attendance file at %s", path)
        except OSError:
            logging.exception("Failed to create attendance file")
            raise

    return path


def _detect_format(path: str) -> str:
    """
    Returns "v1" for [name, datetime] and "v2" for [name,date,punch_in,punch_out].
    Defaults to "v2" if unknown.
    """
    try:
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        return "v2"

    header_lower = [h.strip().lower() for h in header]
    if header_lower == ["name", "datetime"]:
        return "v1"
    if header_lower == ["name", "date", "punch_in", "punch_out"]:
        return "v2"
    return "v2"


def _upgrade_v1_to_v2_in_place(path: str = ATTENDANCE_FILE) -> None:
    """
    If `path` is in v1 format (name,datetime), convert it to v2 format
    (name,date,punch_in,punch_out) and keep a backup of the original.
    """
    if not os.path.isfile(path):
        return
    if _detect_format(path) != "v1":
        return

    # Read v1 rows
    v1_rows: list[dict] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v1_rows.append(row)

    # Backup original v1
    try:
        if not os.path.isfile(ATTENDANCE_V1_BACKUP):
            with open(ATTENDANCE_V1_BACKUP, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "datetime"])
                for row in v1_rows:
                    writer.writerow([row.get("name", ""), row.get("datetime", "")])
            logging.info("Backed up v1 attendance to %s", ATTENDANCE_V1_BACKUP)
    except Exception:
        logging.exception("Failed to create v1 backup; continuing with upgrade anyway")

    # Convert into v2 rows
    existing: dict[tuple[str, str], dict] = {}
    for row in v1_rows:
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
        existing[key] = {"name": name, "date": date, "punch_in": punch_in, "punch_out": ""}

    # Write v2 back to same path
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "date", "punch_in", "punch_out"])
        writer.writeheader()
        for key in sorted(existing.keys(), key=lambda k: (k[0], k[1])):
            writer.writerow(existing[key])

    logging.info("Upgraded %s from v1 to v2 format in-place", path)


def mark_attendance(name: str, path: str = ATTENDANCE_FILE) -> None:
    """
    Append a new attendance record if the same user hasn't been marked
    within the same day.
    """
    if not name:
        raise ValueError("Name for attendance cannot be empty")

    ensure_attendance_file(path)

    today = datetime.now().date()
    already_marked = False

    try:
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_name = row.get("name")
                try:
                    row_dt = datetime.fromisoformat(row.get("datetime", ""))
                except Exception:
                    continue

                if row_name == name and row_dt.date() == today:
                    already_marked = True
                    break
    except FileNotFoundError:
        # Should not happen due to ensure_attendance_file, but handle anyway
        logging.warning("Attendance file not found when reading, recreating")
        ensure_attendance_file(path)

    if already_marked:
        logging.info("Attendance already marked today for %s", name)
        return

    now_str = datetime.now().isoformat(timespec="seconds")
    try:
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, now_str])
        logging.info("Attendance marked for %s at %s", name, now_str)
    except OSError:
        logging.exception("Failed to write to attendance file")
        raise


def punch(name: str, path: str = ATTENDANCE_FILE, cooldown_seconds: int = 15) -> str:
    """
    Punch-in / Punch-out logic.

    - If user has no row for today -> create row with punch_in=now, punch_out=""
    - If user has a row for today and punch_out is empty -> set punch_out=now
    - If user already has punch_out for today -> return "already_done"

    Returns one of:
      "punch_in", "punch_out", "already_done"

    Notes:
    - If `attendance.csv` is still old v1 format, we write to `attendance_v2.csv`.
    """
    if not name:
        raise ValueError("Name for attendance cannot be empty")

    # Always prefer writing to `attendance.csv` by upgrading it if needed.
    _upgrade_v1_to_v2_in_place(path)
    target_path = path
    ensure_attendance_file(target_path)

    today_str = datetime.now().date().isoformat()
    now_str = datetime.now().strftime("%H:%M:%S")

    rows: list[dict] = []
    updated = False

    # Read existing rows
    if os.path.isfile(target_path):
        with open(target_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ["name", "date", "punch_in", "punch_out"]
            for row in reader:
                rows.append(row)
    else:
        fieldnames = ["name", "date", "punch_in", "punch_out"]

    # Find today's row for the user
    for row in rows:
        if row.get("name") == name and row.get("date") == today_str:
            punch_in = (row.get("punch_in") or "").strip()
            punch_out = (row.get("punch_out") or "").strip()

            # Cooldown to prevent immediate double-punch
            try:
                last_time_str = punch_out or punch_in
                if last_time_str:
                    last_dt = datetime.fromisoformat(f"{today_str}T{last_time_str}")
                    delta = (datetime.now() - last_dt).total_seconds()
                    if delta < cooldown_seconds:
                        logging.info("Punch cooldown active for %s (%.1fs < %ds)", name, delta, cooldown_seconds)
                        return "already_done"
            except Exception:
                pass

            if not punch_in:
                row["punch_in"] = now_str
                updated = True
                logging.info("Punch-in for %s at %s", name, now_str)
                action = "punch_in"
            elif not punch_out:
                row["punch_out"] = now_str
                updated = True
                logging.info("Punch-out for %s at %s", name, now_str)
                action = "punch_out"
            else:
                logging.info("Already punched in/out today for %s", name)
                return "already_done"

            break
    else:
        # No row found -> create punch-in row
        rows.append({"name": name, "date": today_str, "punch_in": now_str, "punch_out": ""})
        updated = True
        logging.info("Punch-in (new row) for %s at %s", name, now_str)
        action = "punch_in"

    if updated:
        with open(target_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "date", "punch_in", "punch_out"])
            writer.writeheader()
            writer.writerows(rows)

        return action

    return "already_done"


def main():
    try:
        name = input("Enter name to mark attendance: ").strip()
        mark_attendance(name)
        print(f"Attendance marked for {name}")
    except Exception as e:
        logging.exception("Error in attendance main")
        print(f"Error while marking attendance: {e}")


if __name__ == "__main__":
    main()


