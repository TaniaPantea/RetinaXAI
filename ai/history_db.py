
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retina_predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    image_filename    TEXT    NOT NULL,
    predicted_class   TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    xai_method        TEXT    NOT NULL,
    cam_keep_top_pct  REAL,
    dice              REAL,
    iou               REAL,
    fidelity_top5     REAL,
    fidelity_top10    REAL,
    fidelity_top15    REAL,
    fidelity_top20    REAL,
    stability         REAL,
    stability_corr    REAL,
    deletion_auc      REAL,
    insertion_auc     REAL,
    image_path        TEXT,
    overlay_path      TEXT,
    doctor_mask_path  TEXT,
    doctor_notes      TEXT,
    llm_interpretation TEXT
);
"""

_COLUMNS = [
    "timestamp", "image_filename", "predicted_class", "confidence",
    "xai_method", "cam_keep_top_pct", "dice", "iou",
    "fidelity_top5", "fidelity_top10", "fidelity_top15", "fidelity_top20", 
    "stability", "stability_corr", "deletion_auc", "insertion_auc",
    "image_path", "overlay_path", "doctor_mask_path", "doctor_notes",
    "llm_interpretation",
]

_MIGRATION_COLUMNS = {
    "fidelity_top5": "REAL",
    "fidelity_top10": "REAL",
    "fidelity_top15": "REAL",
    "fidelity_top20": "REAL",
    "stability": "REAL",
    "stability_corr": "REAL",
    "deletion_auc": "REAL",
    "insertion_auc": "REAL",
    "llm_interpretation": "TEXT",
}


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(_SCHEMA)
        existing = {row["name"] for row in
                    conn.execute("PRAGMA table_info(retina_predictions)")}
        for col in _COLUMNS:
            if col not in existing:
                col_type = _MIGRATION_COLUMNS.get(col, "TEXT")
                conn.execute(
                    f"ALTER TABLE retina_predictions ADD COLUMN {col} {col_type}"
                )
        conn.commit()


def insert_prediction(db_path, record: dict) -> int:
    values = [record.get(col) for col in _COLUMNS]
    placeholders = ", ".join("?" for _ in _COLUMNS)
    cols = ", ".join(_COLUMNS)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO retina_predictions ({cols}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)


def update_llm_interpretation(db_path, row_id: int, text: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE retina_predictions SET llm_interpretation = ? WHERE id = ?",
            (text, int(row_id)),
        )
        conn.commit()


def fetch_all(db_path) -> list:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM retina_predictions ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_one(db_path, row_id: int):
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM retina_predictions WHERE id = ?", (row_id,)
        ).fetchone()
    return dict(row) if row else None