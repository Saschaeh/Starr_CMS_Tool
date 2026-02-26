"""
SQLite persistence layer for Starr CMS Content Manager.

Dual-mode:
  - Local dev:  standard sqlite3 with a file at data/starr_cms.db
  - Production: Turso (hosted SQLite) via libsql_experimental

Images are stored as BLOBs in the database (no filesystem dependency).
Set TURSO_DB_URL + TURSO_AUTH_TOKEN env vars to enable Turso mode.
"""

import os

# ---------------------------------------------------------------------------
# Connection setup — detect Turso vs local
# ---------------------------------------------------------------------------

TURSO_DB_URL = os.getenv('TURSO_DB_URL', '')
TURSO_AUTH_TOKEN = os.getenv('TURSO_AUTH_TOKEN', '')
USE_TURSO = bool(TURSO_DB_URL)

if USE_TURSO:
    import libsql_experimental as libsql
else:
    import sqlite3

import threading

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'starr_cms.db')

# Thread-local connection cache (avoids opening/closing per call)
_local = threading.local()


def _rows_to_dicts(cursor):
    """Convert cursor result rows to a list of dicts (works with both drivers)."""
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _row_to_dict(cursor):
    """Convert a single cursor row to a dict, or None."""
    row = cursor.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def get_connection():
    conn = getattr(_local, 'conn', None)
    if conn is not None:
        return conn
    if USE_TURSO:
        conn = libsql.connect(TURSO_DB_URL, auth_token=TURSO_AUTH_TOKEN)
    else:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    return conn


_last_sync_status = ""  # exposed for diagnostics


def _commit(conn):
    """Commit and, for Turso connections, sync to ensure data reaches the remote server."""
    global _last_sync_status
    conn.commit()
    if USE_TURSO:
        if hasattr(conn, 'sync'):
            try:
                conn.sync()
                _last_sync_status = "sync OK"
            except Exception as e:
                _last_sync_status = f"sync FAILED: {e}"
        else:
            _last_sync_status = "no sync() method on connection"
    else:
        _last_sync_status = "local sqlite (no sync needed)"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    # executescript not available in libsql; run statements individually
    stmts = [
        """CREATE TABLE IF NOT EXISTS restaurants (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            website_url TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant TEXT NOT NULL,
            field_name TEXT NOT NULL,
            original_filename TEXT DEFAULT '',
            image_data BLOB,
            alt_text TEXT DEFAULT '',
            overlay_opacity INTEGER DEFAULT 40,
            FOREIGN KEY (restaurant) REFERENCES restaurants(name) ON DELETE CASCADE,
            UNIQUE(restaurant, field_name)
        )""",
        """CREATE TABLE IF NOT EXISTS copy_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant TEXT NOT NULL,
            section_id TEXT NOT NULL,
            content TEXT DEFAULT '',
            FOREIGN KEY (restaurant) REFERENCES restaurants(name) ON DELETE CASCADE,
            UNIQUE(restaurant, section_id)
        )""",
    ]
    for sql in stmts:
        conn.execute(sql)
    # Migrate: add columns if missing (existing databases)
    # Commit each ALTER individually — on Turso/libsql a failed ALTER
    # (column already exists) can taint the transaction, so we commit
    # after each success to keep the connection clean.
    for col, col_def in [
        ('notes', "TEXT DEFAULT ''"),
        ('primary_color', "TEXT DEFAULT ''"),
        ('checklist', "TEXT DEFAULT ''"),
        ('booking_platform', "TEXT DEFAULT ''"),
        ('opentable_rid', "TEXT DEFAULT ''"),
        ('pull_data', "INTEGER DEFAULT 0"),
        ('tripleseat_form_id', "TEXT DEFAULT ''"),
        ('resy_url', "TEXT DEFAULT ''"),
        ('mailing_list_url', "TEXT DEFAULT ''"),
        ('facebook_url', "TEXT DEFAULT ''"),
        ('instagram_url', "TEXT DEFAULT ''"),
        ('phone', "TEXT DEFAULT ''"),
        ('email_general', "TEXT DEFAULT ''"),
        ('email_events', "TEXT DEFAULT ''"),
        ('email_marketing', "TEXT DEFAULT ''"),
        ('email_press', "TEXT DEFAULT ''"),
        ('address', "TEXT DEFAULT ''"),
        ('google_maps_url', "TEXT DEFAULT ''"),
        ('order_online_url', "TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"ALTER TABLE restaurants ADD COLUMN {col} {col_def}")
            _commit(conn)
        except Exception:
            pass  # column already exists


# ─── Restaurant CRUD ─────────────────────────────────────────────────────────

def add_restaurant(name, display_name, website_url=''):
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO restaurants (name, display_name, website_url) VALUES (?, ?, ?)",
        (name, display_name, website_url)
    )
    _commit(conn)


def update_restaurant_url(name, website_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET website_url = ? WHERE name = ?", (website_url, name))
    _commit(conn)


def update_restaurant_color(name, primary_color):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET primary_color = ? WHERE name = ?", (primary_color, name))
    _commit(conn)


def update_restaurant_checklist(name, checklist_json):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET checklist = ? WHERE name = ?", (checklist_json, name))
    _commit(conn)


def update_restaurant_booking(name, booking_platform):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET booking_platform = ? WHERE name = ?", (booking_platform, name))
    _commit(conn)


def update_restaurant_opentable_rid(name, opentable_rid):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET opentable_rid = ? WHERE name = ?", (opentable_rid, name))
    _commit(conn)


def update_restaurant_tripleseat(name, tripleseat_form_id):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET tripleseat_form_id = ? WHERE name = ?", (tripleseat_form_id, name))
    _commit(conn)


def update_restaurant_resy_url(name, resy_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET resy_url = ? WHERE name = ?", (resy_url, name))
    _commit(conn)


def update_restaurant_mailing_list_url(name, mailing_list_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET mailing_list_url = ? WHERE name = ?", (mailing_list_url, name))
    _commit(conn)


def update_restaurant_facebook_url(name, facebook_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET facebook_url = ? WHERE name = ?", (facebook_url, name))
    _commit(conn)


def update_restaurant_instagram_url(name, instagram_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET instagram_url = ? WHERE name = ?", (instagram_url, name))
    _commit(conn)


def update_restaurant_phone(name, phone):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET phone = ? WHERE name = ?", (phone, name))
    _commit(conn)


def update_restaurant_email_general(name, email_general):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET email_general = ? WHERE name = ?", (email_general, name))
    _commit(conn)


def update_restaurant_email_events(name, email_events):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET email_events = ? WHERE name = ?", (email_events, name))
    _commit(conn)


def update_restaurant_email_marketing(name, email_marketing):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET email_marketing = ? WHERE name = ?", (email_marketing, name))
    _commit(conn)


def update_restaurant_email_press(name, email_press):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET email_press = ? WHERE name = ?", (email_press, name))
    _commit(conn)


def update_restaurant_address(name, address):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET address = ? WHERE name = ?", (address, name))
    _commit(conn)


def update_restaurant_google_maps_url(name, google_maps_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET google_maps_url = ? WHERE name = ?", (google_maps_url, name))
    _commit(conn)


def update_restaurant_order_online_url(name, order_online_url):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET order_online_url = ? WHERE name = ?", (order_online_url, name))
    _commit(conn)


def update_restaurant_pull_data(name, pull_data):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET pull_data = ? WHERE name = ?", (int(pull_data), name))
    _commit(conn)


def get_all_restaurants():
    """Return list of dicts with all restaurant columns."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT name, display_name, website_url, notes, primary_color, checklist,"
        " booking_platform, opentable_rid, pull_data, tripleseat_form_id, resy_url,"
        " mailing_list_url, facebook_url, instagram_url, phone, email_general,"
        " email_events, email_marketing, email_press, address, google_maps_url,"
        " order_online_url"
        " FROM restaurants ORDER BY display_name COLLATE NOCASE"
    )
    results = _rows_to_dicts(cur)
    return results


def update_restaurant_notes(name, notes):
    conn = get_connection()
    conn.execute("UPDATE restaurants SET notes = ? WHERE name = ?", (notes, name))
    _commit(conn)


def delete_restaurant(name):
    """Delete restaurant and all associated data from DB."""
    conn = get_connection()
    # Explicit deletes — ON DELETE CASCADE requires PRAGMA foreign_keys=ON
    # which Turso/libsql may not support
    conn.execute("DELETE FROM images WHERE restaurant = ?", (name,))
    conn.execute("DELETE FROM copy_sections WHERE restaurant = ?", (name,))
    conn.execute("DELETE FROM restaurants WHERE name = ?", (name,))
    _commit(conn)


# ─── Image CRUD ──────────────────────────────────────────────────────────────

def save_image(restaurant, field_name, image_bytes, original_filename, alt_text='', overlay_opacity=40):
    """Save processed image bytes as a BLOB in the database."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO images (restaurant, field_name, original_filename, image_data, alt_text, overlay_opacity)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(restaurant, field_name) DO UPDATE SET
            original_filename = excluded.original_filename,
            image_data = excluded.image_data,
            alt_text = excluded.alt_text,
            overlay_opacity = excluded.overlay_opacity
    """, (restaurant, field_name, original_filename, image_bytes, alt_text, overlay_opacity))
    _commit(conn)


def delete_image(restaurant, field_name):
    """Delete a single image record from the database."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM images WHERE restaurant = ? AND field_name = ?",
        (restaurant, field_name)
    )
    _commit(conn)


def update_alt_text(restaurant, field_name, alt_text):
    conn = get_connection()
    conn.execute(
        "UPDATE images SET alt_text = ? WHERE restaurant = ? AND field_name = ?",
        (alt_text, restaurant, field_name)
    )
    _commit(conn)


def update_overlay(restaurant, field_name, overlay_opacity):
    conn = get_connection()
    conn.execute(
        "UPDATE images SET overlay_opacity = ? WHERE restaurant = ? AND field_name = ?",
        (overlay_opacity, restaurant, field_name)
    )
    _commit(conn)


def get_images_for_restaurant(restaurant):
    """Return dict of field_name -> {alt_text, overlay_opacity, original_filename, has_image}.
    Does NOT return image_data to avoid loading all blobs into memory at once."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT field_name, alt_text, overlay_opacity, original_filename, "
        "(CASE WHEN image_data IS NOT NULL THEN 1 ELSE 0 END) AS has_image "
        "FROM images WHERE restaurant = ?",
        (restaurant,)
    )
    rows = _rows_to_dicts(cur)
    return {r['field_name']: r for r in rows}


def get_image_data(restaurant, field_name):
    """Return the raw image bytes for a single field, or None."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT image_data FROM images WHERE restaurant = ? AND field_name = ?",
        (restaurant, field_name)
    )
    row = _row_to_dict(cur)
    if row and row['image_data']:
        return bytes(row['image_data'])
    return None


def get_image_record(restaurant, field_name):
    """Return metadata (no blob) for a single image field."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT field_name, alt_text, overlay_opacity, original_filename "
        "FROM images WHERE restaurant = ? AND field_name = ?",
        (restaurant, field_name)
    )
    result = _row_to_dict(cur)
    return result


# ─── Copy CRUD ────────────────────────────────────────────────────────────────

def save_copy_section(restaurant, section_id, content):
    conn = get_connection()
    conn.execute("""
        INSERT INTO copy_sections (restaurant, section_id, content)
        VALUES (?, ?, ?)
        ON CONFLICT(restaurant, section_id) DO UPDATE SET content = excluded.content
    """, (restaurant, section_id, content))
    _commit(conn)


def get_copy_for_restaurant(restaurant):
    """Return dict of section_id -> content."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT section_id, content FROM copy_sections WHERE restaurant = ?",
        (restaurant,)
    )
    rows = _rows_to_dicts(cur)
    return {r['section_id']: r['content'] for r in rows}


def save_all_copy(restaurant, copy_dict):
    """Save multiple copy sections at once."""
    conn = get_connection()
    for section_id, content in copy_dict.items():
        conn.execute("""
            INSERT INTO copy_sections (restaurant, section_id, content)
            VALUES (?, ?, ?)
            ON CONFLICT(restaurant, section_id) DO UPDATE SET content = excluded.content
        """, (restaurant, section_id, content))
    _commit(conn)
