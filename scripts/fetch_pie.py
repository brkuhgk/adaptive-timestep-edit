"""Download PIE-Bench from the Google Drive folder link in your access email, 8 files at a time.

    python scripts/fetch_pie.py "<Drive folder link>"     # -> datasets/pie_download (re-run to resume)

`gdown --folder` alone fetches one file at a time (~1.5 s/file); this lists the folder once, then downloads in
parallel and re-tries any file that is missing, empty, or not a readable image.
"""
import concurrent.futures as cf
import os
import sys

import gdown
from PIL import Image

url = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "datasets/pie_download"
files = gdown.download_folder(url, output=out, skip_download=True, quiet=True)
print("listed", len(files), flush=True)


def ok(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    if p.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            with Image.open(p) as im:
                im.load()
        except Exception:
            return False
    return True


todo = [f for f in files if not ok(f.local_path)]
print("to download", len(todo), flush=True)


def dl(f):
    os.makedirs(os.path.dirname(f.local_path), exist_ok=True)
    for _ in range(3):
        try:
            gdown.download(id=f.id, output=f.local_path, quiet=True)
            if ok(f.local_path):
                return 1
        except Exception:
            pass
    print("FAILED", f.local_path, flush=True)
    return 0


with cf.ThreadPoolExecutor(8) as ex:
    n = sum(ex.map(dl, todo))
print("downloaded", n, "of", len(todo), flush=True)
bad = [f.local_path for f in files if not ok(f.local_path)]
print("still bad", len(bad), bad[:5], flush=True)
sys.exit(1 if bad else 0)
