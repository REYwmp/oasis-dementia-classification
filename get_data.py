"""Download the pinned public copy used for the renewal and check its contents."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

SOURCE = ("https://raw.githubusercontent.com/NizarIslah/dementia-prediction/"
          "de76175f53465fa0167dceb8cf57aabf717005c0/oasis_longitudinal.csv")
EXPECTED = "23d85403b796da00829c143b2268c0ca31656630562791ec7b1c3dac4c0405d5"


def verified(data):
    normalized = data.decode("utf-8-sig").replace("\r\n", "\n").strip().encode()
    return hashlib.sha256(normalized).hexdigest() == EXPECTED


if __name__ == "__main__":
    dest = Path("data/oasis_longitudinal.csv")
    if dest.exists():
        if not verified(dest.read_bytes()):
            raise SystemExit(f"{dest} differs from the archived source. Supply it directly to analyze.py or move it before downloading.")
    else:
        with urlopen(SOURCE, timeout=45) as response:
            data = response.read()
        if not verified(data):
            raise SystemExit("Download did not match the archived CSV checksum.")
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(data)
    print(f"Verified {dest}")
