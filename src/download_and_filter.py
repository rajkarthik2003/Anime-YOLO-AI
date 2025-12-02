import os
import requests
import pandas as pd
from tqdm import tqdm

# 1) ALWAYS use all_data.csv from project root
CSV_PATH = "all_data.csv"
OUT_IMG_DIR = os.path.join("data", "raw", "images")
MANIFEST_CSV = os.path.join("data", "raw", "manifest_filtered.csv")

# 5) Classes used to filter by substring match within 'tags'
CLASSES = ["naruto", "luffy", "gojo", "goku", "sukuna"]

os.makedirs(OUT_IMG_DIR, exist_ok=True)


def normalize_url(url: str) -> str:
    """4) Prefix with https: if URL starts with '//'."""
    if not url:
        return ""
    url = str(url).strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return ""


def tags_match(tags_str: str) -> bool:
    """5) Return True if any class name appears in the tags string (case-insensitive)."""
    if not tags_str or pd.isna(tags_str):
        return False
    t = str(tags_str).lower()
    return any(cls in t for cls in CLASSES)


def download_with_retries(url: str, out_path: str, attempts: int = 3, timeout: int = 20) -> bool:
    """7) Download to out_path with up to 3 retries. Returns True on success."""
    if not url:
        return False
    for _ in range(attempts):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
    return False


def main():
    # 1) Load all_data.csv
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")

    # 2) Detect tags column (must be 'tags')
    if "tags" not in df.columns:
        raise KeyError(
            f"'tags' column not found in {CSV_PATH}. Available columns: {list(df.columns)}"
        )
    print("Using tags column: tags")

    # 3) Detect image URL column (must be 'sample_url')
    if "sample_url" not in df.columns:
        raise KeyError(
            f"'sample_url' column not found in {CSV_PATH}. Available columns: {list(df.columns)}"
        )
    print("Using URL column: sample_url")

    # 5) Filter rows where tags contain ANY of the classes
    df_matched = df[df["tags"].apply(tags_match)]
    matched_count = len(df_matched)
    print(f"Matched rows: {matched_count}")

    # 6) Download each valid image into data/raw/images/
    downloaded_records = []
    for _, row in tqdm(df_matched.iterrows(), total=matched_count, desc="Downloading"):
        # 10) Skip rows with NULL URLs or missing tags
        raw_url = row.get("sample_url", "")
        if pd.isna(raw_url) or not raw_url:
            continue
        url = normalize_url(raw_url)
        if not url:
            continue

        # filename: id + basename of URL (without query)
        idx = row.get("id", row.get("post_id", "unknown"))
        base_name = os.path.basename(url.split("?")[0])
        fn = f"{idx}_{base_name}" if base_name else f"{idx}.jpg"
        out_path = os.path.join(OUT_IMG_DIR, fn)

        if os.path.exists(out_path):
            downloaded = True
        else:
            downloaded = download_with_retries(url, out_path)

        if downloaded:
            rec = row.to_dict()
            rec["file"] = out_path
            downloaded_records.append(rec)

    # 8) Create manifest_filtered.csv containing only matched + downloaded rows
    manifest_df = pd.DataFrame(downloaded_records)
    os.makedirs(os.path.dirname(MANIFEST_CSV), exist_ok=True)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    # 9) Print number of matched rows and successfully downloaded images
    print(f"Matched rows: {matched_count}")
    print(f"Successfully downloaded images: {len(manifest_df)}")
    print(f"Manifest written to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
