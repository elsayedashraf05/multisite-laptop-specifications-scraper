"""
=============================================================================
  Laptop Dataset Cleaning Pipeline
  Senior Data Engineer — Production-Ready Script
=============================================================================
  Input  : all_laptops.csv   (1 008 rows, scraped from Amazon/Noon/Jumia)
  Output : final_ready_laptops.csv

  Pipeline stages
  ───────────────
  1.  Load & audit
  2.  Drop rows missing price or name
  3.  Fill rating / reviews_count with 0
  4.  Clean & validate  price  → float EGP
  5.  Clean  screen_size       → float inches
  6.  Clean & swap  ram / storage  (fix cross-bleed errors)
  7.  Fill missing specs from the name column via Regex
  8.  Standardise  processor   → canonical family strings
  9.  Standardise  gpu         → NVIDIA / AMD / Intel families
  10. Fill remaining spec NaN  → "Not Specified"
  11. Cross-site deduplication → keep lowest price per (brand,proc,ram,storage)
  12. Final column typing & ordering
  13. Save  final_ready_laptops.csv  (utf-8-sig)
  14. Print summary report
=============================================================================
"""

import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = "/mnt/user-data/uploads/all_laptops.csv"
OUTPUT_FILE = "/mnt/user-data/outputs/final_ready_laptops.csv"

# RAM values that are clearly storage capacities mis-placed in the RAM column
_STORAGE_LOOKING_RAM = {512, 256, 128, 1000, 2000, 2050}   # GB values > realistic RAM

# Realistic RAM upper-bound (GB).  Anything above is treated as an error.
_MAX_REASONABLE_RAM_GB = 128


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD & AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Read the CSV file with UTF-8 BOM encoding (handles Arabic text),
    then print a quick audit so the engineer can see the raw state.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    print("=" * 65)
    print("  AUDIT — RAW DATA")
    print("=" * 65)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Sources      : {df['source'].value_counts().to_dict()}")
    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    for col, n in missing[missing > 0].items():
        print(f"    {col:<18} {n:>4}  ({n/len(df)*100:.1f}%)")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DROP CRITICAL NULLS
# ─────────────────────────────────────────────────────────────────────────────

def drop_critical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where 'name' or 'price' is missing because they cannot
    be used for price-comparison or dashboard display.
    """
    before = len(df)
    df = df.dropna(subset=["name", "price"]).copy()
    after = len(df)
    print(f"[Stage 2] Dropped {before - after} rows with missing name/price. "
          f"Remaining: {after:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — FILL RATING & REVIEWS_COUNT
# ─────────────────────────────────────────────────────────────────────────────

def fill_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Products scraped from Jumia have no rating/review data.
    Fill with 0 so numeric aggregations don't fail downstream.
    """
    df["rating"]        = df["rating"].fillna(0).astype(float)
    df["reviews_count"] = df["reviews_count"].fillna(0).astype(int)
    print(f"[Stage 3] rating / reviews_count nulls filled with 0.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — PRICE
# ─────────────────────────────────────────────────────────────────────────────

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure price is a clean float in EGP.
    The previous pipeline already converted commas; this stage is a safety net
    that strips any residual currency symbols or whitespace, then coerces.
    """
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)   # keep only digits and dot
        .replace("", np.nan)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop the tiny number of rows where price couldn't be parsed
    before = len(df)
    df = df.dropna(subset=["price"]).copy()
    print(f"[Stage 4] Price cleaned → float EGP.  "
          f"Dropped {before - len(df)} unparseable rows.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — SCREEN SIZE
# ─────────────────────────────────────────────────────────────────────────────

def clean_screen_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip trailing quotes / text (e.g. '15.6"') and convert to float inches.
    Values that don't resolve to a plausible laptop screen (10–21 inches)
    are replaced with NaN.

    Known edge-case: one row has '313' because the model number 'MRXQ3' was
    fused with '13-inch' during scraping.  This is caught by the range check.
    """
    def _parse_screen(val):
        if pd.isna(val):
            return np.nan
        # Remove all non-numeric characters except the decimal point
        cleaned = re.sub(r'[^\d.]', '', str(val))
        try:
            f = float(cleaned)
            # Plausible laptop screen: 10 – 21 inches
            return f if 10.0 <= f <= 21.0 else np.nan
        except ValueError:
            return np.nan

    df["screen_size"] = df["screen_size"].apply(_parse_screen)
    print(f"[Stage 5] screen_size → float inches  "
          f"({df['screen_size'].notna().sum():,} valid values).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — RAM / STORAGE CROSS-BLEED CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gb_value(text: str) -> float | None:
    """Return the leading numeric value (GB) from a string like '16GB'."""
    if pd.isna(text):
        return None
    m = re.match(r"(\d+(?:\.\d+)?)", str(text).strip())
    return float(m.group(1)) if m else None


def _normalise_storage(raw: str) -> str:
    """
    Return a canonical storage string like '512GB SSD', '1TB HDD', or keep
    the value as-is if already correct.
    """
    if pd.isna(raw):
        return raw
    raw = str(raw).strip()
    # Already has a type label
    if re.search(r'(SSD|HDD|NVMe)', raw, re.I):
        # Standardise spacing and capitalisation
        raw = re.sub(r'\s+', ' ', raw).upper()
        raw = re.sub(r'NVME', 'SSD', raw)       # normalise NVMe → SSD label
        return raw.strip()
    # No label — just a size
    return raw


def clean_ram_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct three categories of RAM / storage cross-bleed errors:

    Case A — RAM contains a storage-sized value (e.g. '512GB', '2050GB'):
        • If storage is missing or tiny (<= 64 GB without a type),
          swap the values.
        • Otherwise mark RAM as NaN so the regex stage can re-extract it.

    Case B — Storage contains a tiny implausible value (e.g. '1GB SSD',
              '6GB SSD', '8GB SSD', '16GB SSD', '32GB SSD') that matches
              the RAM value:
        • The storage value is really a RAM bleed; re-extract storage from
          the product name.

    Case C — Both RAM and storage look correct → leave untouched.
    """
    swap_count  = 0
    clear_count = 0

    for idx, row in df.iterrows():
        ram_val  = str(row["ram"]).strip()   if pd.notna(row["ram"])     else None
        stor_val = str(row["storage"]).strip() if pd.notna(row["storage"]) else None

        # ── Case A: suspicious RAM value ────────────────────────────────────
        if ram_val:
            gb = _extract_gb_value(ram_val)
            if gb is not None and (gb > _MAX_REASONABLE_RAM_GB or
                                   gb in _STORAGE_LOOKING_RAM):
                # Only swap if the current storage is missing or also bad
                stor_gb = _extract_gb_value(stor_val) if stor_val else None
                stor_is_tiny = (
                    stor_gb is not None
                    and stor_gb <= 64
                    and not re.search(r'(SSD|HDD)', stor_val or '', re.I)
                )
                if stor_val is None or stor_is_tiny:
                    # Swap
                    df.at[idx, "ram"]     = stor_val   # may be None → NaN
                    df.at[idx, "storage"] = _normalise_storage(ram_val)
                    swap_count += 1
                else:
                    # Can't determine correct RAM — clear it for regex re-extract
                    df.at[idx, "ram"] = np.nan
                    clear_count += 1

        # ── Case B: suspicious storage value (tiny) ─────────────────────────
        # Re-check after potential swap above
        stor_val = str(df.at[idx, "storage"]).strip() if pd.notna(df.at[idx, "storage"]) else None
        ram_val  = str(df.at[idx, "ram"]).strip()      if pd.notna(df.at[idx, "ram"])     else None
        if stor_val:
            stor_gb = _extract_gb_value(stor_val)
            if stor_gb is not None and stor_gb <= 64 and re.search(r'SSD|HDD', stor_val, re.I):
                # Is this value the same as RAM? If so, it's a RAM bleed into storage
                ram_gb = _extract_gb_value(ram_val) if ram_val else None
                if ram_gb is not None and abs(stor_gb - ram_gb) < 1:
                    # The storage field mirrors RAM — wipe it for re-extraction
                    df.at[idx, "storage"] = np.nan
                    clear_count += 1

    print(f"[Stage 6] RAM/Storage cross-bleed: "
          f"{swap_count} swaps, {clear_count} cleared for re-extraction.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — REGEX EXTRACTION FROM name FOR MISSING SPECS
# ─────────────────────────────────────────────────────────────────────────────

# ── Compiled patterns ──────────────────────────────────────────────────────

_RE_RAM = re.compile(
    r'(?:RAM\s*)?(\d+)\s*G(?:B)?\s*(?:RAM|DDR[345X]?|LPDDR\w*)'  # 16GB DDR5
    r'|(\d+)\s*GB\s*RAM'                                            # 16GB RAM
    r'|RAM\s*(\d+)\s*G',                                            # RAM 16G
    re.I
)

_RE_STORAGE = re.compile(
    r'(\d+)\s*(TB|GB)\s*(?:[A-Z0-9.]+\s*)?(?:SSD|HDD|NVMe|NVME)',  # 512GB NVMe SSD
    re.I
)

_RE_SCREEN = re.compile(
    r'(\d{1,2}(?:\.\d)?)\s*[-–]?\s*inch(?:es)?'     # 15.6 inch / 15.6-inch
    r'|(\d{1,2}(?:\.\d)?)\s*["\u201c\u201d]',        # 15.6"
    re.I
)

_RE_PROC = [
    re.compile(r'(Apple\s+M[1-5](?:\s+(?:Pro|Max|Ultra|Air))?)', re.I),
    re.compile(r'(Intel\s+Core\s+Ultra\s+[579]\s+[\w]+)', re.I),
    re.compile(r'(Intel\s+Core\s+[iI][3579][-\s]\d{4,5}[A-Z]*)', re.I),
    re.compile(r'(Intel\s+Core\s+[iI][3579])(?=[\s,\-/|])', re.I),
    re.compile(r'(Intel\s+Core\s+[5-9]\s+\d{3,6}[A-Z]*)', re.I),
    re.compile(r'(Intel\s+Core\s+Ultra\s+\d+)', re.I),
    re.compile(r'(AMD\s+Ryzen\s+(?:AI\s+)?[3579]\s+\w+)', re.I),
    re.compile(r'(AMD\s+Ryzen\s+AI\s+MAX\+?\s*\d*)', re.I),
    re.compile(r'(Ryzen\s+[3579]\s+[\w]+)', re.I),
    re.compile(r'(Intel\s+Celeron\s+\w+)', re.I),
    re.compile(r'(Intel\s+Pentium\s+\w+)', re.I),
    re.compile(r'(Snapdragon\s+X(?:\s+\w+)?)', re.I),
    re.compile(r'(Intel\s+N\d{3,4})', re.I),
    re.compile(r'(AMD\s+A\d[-\s]\d+\w*)', re.I),
]

_RE_GPU = [
    re.compile(r'(NVIDIA\s+GeForce\s+RTX\s*\d{4}\s*(?:Ti|Super)?)', re.I),
    re.compile(r'(NVIDIA\s+GeForce\s+GTX\s*\d{4}\s*(?:Ti)?)', re.I),
    re.compile(r'(NVIDIA\s+GeForce\s+MX\s*\d+)', re.I),
    re.compile(r'(GeForce\s+RTX\s*\d{4}\s*(?:Ti|Super)?)', re.I),
    re.compile(r'(RTX\s*\d{4}\s*(?:Ti|Super)?)', re.I),
    re.compile(r'(GTX\s*\d{4}\s*(?:Ti)?)', re.I),
    re.compile(r'(AMD\s+Radeon\s+RX\s*\d{4}[MS]?)', re.I),
    re.compile(r'(AMD\s+Radeon\s+(?:Graphics|\d+\w*))', re.I),
    re.compile(r'(Radeon\s+R[0-9]\s+Graphics)', re.I),
    re.compile(r'(Intel\s+Arc\s+\w+)', re.I),
    re.compile(r'(Intel\s+Iris\s+Xe\s*(?:Graphics)?)', re.I),
    re.compile(r'(Intel\s+Iris\s+Plus\s*(?:Graphics)?)', re.I),
    re.compile(r'(Intel\s+UHD\s+(?:Graphics\s*)?\d*)', re.I),
    re.compile(r'(Intel\s+HD\s+Graphics\s*\d*)', re.I),
]


def _extract_ram_from_name(name: str) -> str | None:
    m = _RE_RAM.search(name)
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        gb  = float(val)
        if 1 <= gb <= _MAX_REASONABLE_RAM_GB:
            return f"{int(gb)}GB"
    return None


def _extract_storage_from_name(name: str) -> str | None:
    m = _RE_STORAGE.search(name)
    if m:
        size, unit = m.group(1), m.group(2).upper()
        stype = "HDD" if re.search(r'HDD', m.group(0), re.I) else "SSD"
        return f"{size}{unit} {stype}"
    return None


def _extract_screen_from_name(name: str) -> float | None:
    m = _RE_SCREEN.search(name)
    if m:
        val = m.group(1) or m.group(2)
        try:
            f = float(val)
            return f if 10.0 <= f <= 21.0 else None
        except ValueError:
            return None
    return None


def _extract_processor_from_name(name: str) -> str | None:
    for pat in _RE_PROC:
        m = pat.search(name)
        if m:
            raw = re.sub(r'\s+', ' ', m.group(1)).strip()
            return raw
    return None


def _extract_gpu_from_name(name: str) -> str | None:
    for pat in _RE_GPU:
        m = pat.search(name)
        if m:
            raw = re.sub(r'\s+', ' ', m.group(1)).strip()
            return raw
    return None


def extract_missing_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every spec column that is still NaN, attempt Regex extraction from
    the product 'name'.  Applies to: ram, storage, screen_size,
    processor, gpu.
    """
    specs_filled = {col: 0 for col in
                    ["ram", "storage", "screen_size", "processor", "gpu"]}

    for idx, row in df.iterrows():
        name = str(row["name"])

        if pd.isna(row["ram"]):
            val = _extract_ram_from_name(name)
            if val:
                df.at[idx, "ram"] = val
                specs_filled["ram"] += 1

        if pd.isna(row["storage"]):
            val = _extract_storage_from_name(name)
            if val:
                df.at[idx, "storage"] = val
                specs_filled["storage"] += 1

        if pd.isna(row["screen_size"]):
            val = _extract_screen_from_name(name)
            if val:
                df.at[idx, "screen_size"] = val
                specs_filled["screen_size"] += 1

        if pd.isna(row["processor"]):
            val = _extract_processor_from_name(name)
            if val:
                df.at[idx, "processor"] = val
                specs_filled["processor"] += 1

        if pd.isna(row["gpu"]):
            val = _extract_gpu_from_name(name)
            if val:
                df.at[idx, "gpu"] = val
                specs_filled["gpu"] += 1

    print("[Stage 7] Specs recovered from name column:")
    for col, n in specs_filled.items():
        print(f"    {col:<14}  +{n} values filled")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — STANDARDISE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

# Mapping: (regex to match) → canonical family label
_PROC_FAMILY_MAP = [
    # Apple
    (re.compile(r'apple\s+m1\b', re.I),                 "Apple M1"),
    (re.compile(r'apple\s+m2\b', re.I),                 "Apple M2"),
    (re.compile(r'apple\s+m3\b', re.I),                 "Apple M3"),
    (re.compile(r'apple\s+m4\s+pro', re.I),             "Apple M4 Pro"),
    (re.compile(r'apple\s+m4\b', re.I),                 "Apple M4"),
    (re.compile(r'apple\s+m5\b', re.I),                 "Apple M5"),
    # Intel Core Ultra
    (re.compile(r'intel\s+core\s+ultra\s+9', re.I),     "Intel Core Ultra 9"),
    (re.compile(r'intel\s+core\s+ultra\s+7', re.I),     "Intel Core Ultra 7"),
    (re.compile(r'intel\s+core\s+ultra\s+5', re.I),     "Intel Core Ultra 5"),
    (re.compile(r'ultra\s+9', re.I),                    "Intel Core Ultra 9"),
    (re.compile(r'ultra\s+7', re.I),                    "Intel Core Ultra 7"),
    (re.compile(r'ultra\s+5', re.I),                    "Intel Core Ultra 5"),
    # Intel Core i-series
    (re.compile(r'core\s+i9', re.I),                    "Intel Core i9"),
    (re.compile(r'core\s+i7', re.I),                    "Intel Core i7"),
    (re.compile(r'core\s+i5', re.I),                    "Intel Core i5"),
    (re.compile(r'core\s+i3', re.I),                    "Intel Core i3"),
    # Intel Core (no i-prefix, newer naming)
    (re.compile(r'intel\s+core\s+[79]\b', re.I),        "Intel Core 7/9"),
    (re.compile(r'intel\s+core\s+5\b', re.I),           "Intel Core 5"),
    # AMD Ryzen
    (re.compile(r'ryzen\s+ai\s+max', re.I),             "AMD Ryzen AI MAX"),
    (re.compile(r'ryzen\s+(?:ai\s+)?9', re.I),          "AMD Ryzen 9"),
    (re.compile(r'ryzen\s+(?:ai\s+)?7', re.I),          "AMD Ryzen 7"),
    (re.compile(r'ryzen\s+(?:ai\s+)?5', re.I),          "AMD Ryzen 5"),
    (re.compile(r'ryzen\s+(?:ai\s+)?3', re.I),          "AMD Ryzen 3"),
    # Low-power / other Intel
    (re.compile(r'intel\s+celeron', re.I),               "Intel Celeron"),
    (re.compile(r'intel\s+pentium', re.I),               "Intel Pentium"),
    (re.compile(r'intel\s+n\d{3,4}', re.I),             "Intel N-series"),
    # Snapdragon
    (re.compile(r'snapdragon', re.I),                    "Snapdragon X"),
    # AMD A-series
    (re.compile(r'amd\s+a[0-9]', re.I),                 "AMD A-series"),
]


def standardise_processor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the raw processor string to a clean canonical family label.
    The original (detailed) string is preserved in 'processor_detail'.
    """
    df["processor_detail"] = df["processor"].copy()   # keep full detail

    def _categorise(raw):
        if pd.isna(raw):
            return np.nan
        for pat, label in _PROC_FAMILY_MAP:
            if pat.search(str(raw)):
                return label
        return str(raw).strip()   # return cleaned original if no match

    df["processor"] = df["processor"].apply(_categorise)

    families = df["processor"].dropna().value_counts()
    print(f"[Stage 8] Processor families after standardisation "
          f"({families.shape[0]} unique):")
    for fam, cnt in families.head(12).items():
        print(f"    {fam:<25}  {cnt:>4}")
    if families.shape[0] > 12:
        print(f"    ... and {families.shape[0] - 12} more families")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — STANDARDISE GPU
# ─────────────────────────────────────────────────────────────────────────────

_GPU_FAMILY_MAP = [
    # NVIDIA RTX 50-series
    (re.compile(r'RTX\s*50[0-9]{2}', re.I),             "NVIDIA RTX 50-series"),
    # NVIDIA RTX 40-series
    (re.compile(r'RTX\s*40[0-9]{2}', re.I),             "NVIDIA RTX 40-series"),
    # NVIDIA RTX 30-series
    (re.compile(r'RTX\s*30[0-9]{2}', re.I),             "NVIDIA RTX 30-series"),
    # NVIDIA RTX 20-series
    (re.compile(r'RTX\s*20[0-9]{2}', re.I),             "NVIDIA RTX 20-series"),
    # NVIDIA MX-series
    (re.compile(r'MX\s*\d{3}', re.I),                   "NVIDIA GeForce MX"),
    # NVIDIA GTX
    (re.compile(r'GTX\s*\d{4}', re.I),                  "NVIDIA GTX"),
    # AMD Radeon RX
    (re.compile(r'Radeon\s+RX\s*\d{4}', re.I),          "AMD Radeon RX"),
    # AMD Radeon (integrated)
    (re.compile(r'AMD\s+Radeon\s+(?:Graphics|R[0-9]|8\d{3})', re.I), "AMD Radeon Integrated"),
    (re.compile(r'Radeon\s+R[0-9]\s+Graphics', re.I),   "AMD Radeon Integrated"),
    # Intel Arc
    (re.compile(r'Intel\s+Arc', re.I),                   "Intel Arc"),
    # Intel Iris Xe
    (re.compile(r'Intel\s+Iris\s+Xe', re.I),             "Intel Iris Xe"),
    # Intel Iris Plus
    (re.compile(r'Intel\s+Iris\s+Plus', re.I),           "Intel Iris Plus"),
    # Intel UHD
    (re.compile(r'Intel\s+UHD', re.I),                   "Intel UHD Graphics"),
    # Intel HD
    (re.compile(r'Intel\s+HD\s+Graphics', re.I),         "Intel HD Graphics"),
]


def standardise_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw GPU strings to tidy family labels.
    Original string is kept in 'gpu_detail'.
    """
    df["gpu_detail"] = df["gpu"].copy()

    def _categorise(raw):
        if pd.isna(raw):
            return np.nan
        for pat, label in _GPU_FAMILY_MAP:
            if pat.search(str(raw)):
                return label
        return str(raw).strip()

    df["gpu"] = df["gpu"].apply(_categorise)

    families = df["gpu"].dropna().value_counts()
    print(f"[Stage 9] GPU families after standardisation "
          f"({families.shape[0]} unique):")
    for fam, cnt in families.head(10).items():
        print(f"    {fam:<30}  {cnt:>4}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10 — FILL REMAINING NULLS WITH "Not Specified"
# ─────────────────────────────────────────────────────────────────────────────

def fill_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    After all extraction attempts, replace any remaining NaN in spec columns
    with the sentinel string "Not Specified".
    Numeric columns (price, rating, reviews_count, screen_size) keep NaN.
    """
    spec_cols = ["processor", "processor_detail",
                 "ram", "storage", "gpu", "gpu_detail", "model_name"]
    for col in spec_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Not Specified")

    print(f"[Stage 10] Remaining spec NaNs filled with 'Not Specified'.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 11 — CROSS-SITE DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_for_dedup(val: str) -> str:
    """Lower-case, strip whitespace, and collapse spaces for safe comparison."""
    if pd.isna(val) or str(val).strip().lower() in ("not specified", "nan", ""):
        return "__unknown__"
    return re.sub(r'\s+', ' ', str(val).strip().lower())


def deduplicate_cross_site(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the same physical laptop listed on multiple e-commerce sites
    and keep only the listing with the **lowest price** (best for consumers).

    Deduplication key  =  (brand, processor_family, ram, storage)

    Rows where any key field is unknown are excluded from deduplication
    to avoid collapsing unrelated products.
    """
    before = len(df)

    # Build normalised key columns (temporary, not saved to output)
    df["_key_brand"]     = df["brand"].apply(_normalise_for_dedup)
    df["_key_processor"] = df["processor"].apply(_normalise_for_dedup)
    df["_key_ram"]       = df["ram"].apply(_normalise_for_dedup)
    df["_key_storage"]   = df["storage"].apply(_normalise_for_dedup)

    key_cols = ["_key_brand", "_key_processor", "_key_ram", "_key_storage"]

    # Mask rows that have at least one unknown key — cannot safely dedup these
    has_unknown = (df[key_cols] == "__unknown__").any(axis=1)
    dedup_df    = df[~has_unknown].copy()
    skip_df     = df[has_unknown].copy()

    # Within the deduplicated pool: sort by price ascending, keep first
    dedup_df = (
        dedup_df
        .sort_values("price", ascending=True)
        .drop_duplicates(subset=key_cols, keep="first")
    )

    # Recombine
    df = pd.concat([dedup_df, skip_df], ignore_index=True)

    # Drop helper columns
    df.drop(columns=key_cols, inplace=True)

    after = len(df)
    print(f"[Stage 11] Cross-site deduplication: "
          f"{before} → {after} rows  ({before - after} duplicates removed).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 12 — FINAL COLUMN ORDERING & TYPE ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

FINAL_COLUMN_ORDER = [
    "name",
    "brand",
    "model_name",
    "source",
    "price",
    "rating",
    "reviews_count",
    "screen_size",
    "processor",
    "processor_detail",
    "ram",
    "storage",
    "gpu",
    "gpu_detail",
    "url",
    "image",
]


def finalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce column order and correct dtypes.
    """
    # Keep only known final columns (drop any temp/extra columns)
    existing = [c for c in FINAL_COLUMN_ORDER if c in df.columns]
    df = df[existing].copy()

    # Type enforcement
    df["price"]         = df["price"].astype(float).round(2)
    df["rating"]        = df["rating"].astype(float)
    df["reviews_count"] = df["reviews_count"].astype(int)
    df["screen_size"]   = pd.to_numeric(df["screen_size"], errors="coerce")

    # Consistent title-case on brand
    df["brand"] = df["brand"].str.strip().str.title()

    print(f"[Stage 12] Schema finalised.  Columns: {existing}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 13 — SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_output(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned dataframe as a UTF-8 BOM CSV for Excel compatibility."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Stage 13] Saved → {path}  ({len(df):,} rows, "
          f"{df.shape[1]} columns)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Execute all 13 stages in sequence and return the final dataframe.
    """
    print("\n" + "=" * 65)
    print("  LAPTOP DATASET CLEANING PIPELINE — START")
    print("=" * 65 + "\n")

    df = load_data(input_path)
    rows_before = len(df)

    df = drop_critical_nulls(df)
    df = fill_ratings(df)
    df = clean_price(df)
    df = clean_screen_size(df)
    df = clean_ram_storage(df)
    df = extract_missing_specs(df)
    df = standardise_processor(df)
    df = standardise_gpu(df)
    df = fill_remaining_nulls(df)
    df = deduplicate_cross_site(df)
    df = finalise_schema(df)
    save_output(df, output_path)

    rows_after = len(df)

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"  Rows before cleaning        : {rows_before:>6,}")
    print(f"  Rows after cleaning         : {rows_after:>6,}")
    print(f"  Rows removed                : {rows_before - rows_after:>6,}")
    print(f"  Reduction                   : {(rows_before - rows_after)/rows_before*100:>6.1f}%")
    print()
    print("  Column fill rates (final dataset):")
    total = len(df)
    for col in ["price", "screen_size", "processor", "ram", "storage", "gpu",
                "rating", "reviews_count"]:
        if col in df.columns:
            if df[col].dtype == object:
                filled = (df[col] != "Not Specified").sum()
            else:
                filled = df[col].notna().sum()
            pct = filled / total * 100
            print(f"    {col:<20}  {filled:>4}/{total:>4}  ({pct:>5.1f}%)")
    print()
    print(f"  Sources in final dataset:")
    for src, cnt in df["source"].value_counts().items():
        print(f"    {src:<12}  {cnt:>4} rows")
    print()
    print(f"  Output file  : {output_path}")
    print("=" * 65 + "\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    final_df = run_pipeline(INPUT_FILE, OUTPUT_FILE)
