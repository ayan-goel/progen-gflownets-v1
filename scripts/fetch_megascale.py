"""Fetch the relevant member of the pinned Zenodo archive, with resumable ranges.

The published archive is 1 GB; this CSV occupies 210 MB compressed. The offsets,
CRC and size below were read from that archive's ZIP central directory. We verify
the directory itself on every invocation before trusting the pinned member.
"""

import argparse
import hashlib
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from tempfile import TemporaryFile

URL = (
    "https://zenodo.org/api/records/7992926/files/Processed_K50_dG_datasets.zip/content"
)
ARCHIVE_SIZE = 1013678473
MEMBER_NAME = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
OFFSET = 298
MEMBER_BYTES = 209854775
COMPRESSED_BYTES = 209854627
UNCOMPRESSED_BYTES = 697658024
EXPECTED_CRC = 0x9AC87977


def request_range(start, end):
    """Fail if a server ignores byte ranges; never append a full archive to a part."""
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    response = urllib.request.urlopen(request, timeout=60)
    expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise RuntimeError("Server returned an unexpected byte range")
    return response


def fetch(destination):
    """Resume compressed bytes, verify ZIP CRC/length, and atomically publish the CSV."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / Path(MEMBER_NAME).name
    if csv_path.exists():
        raise FileExistsError(
            f"Already exists: {csv_path}; existing data is never overwritten"
        )
    with request_range(ARCHIVE_SIZE - 65536, ARCHIVE_SIZE - 1) as response:
        tail = response.read()
    with TemporaryFile() as sparse_archive:
        sparse_archive.seek(ARCHIVE_SIZE - len(tail))
        sparse_archive.write(tail)
        sparse_archive.flush()
        with zipfile.ZipFile(sparse_archive) as archive:
            member = archive.getinfo(MEMBER_NAME)
            actual = (
                member.header_offset,
                member.compress_size,
                member.file_size,
                member.CRC,
            )
            if actual != (OFFSET, COMPRESSED_BYTES, UNCOMPRESSED_BYTES, EXPECTED_CRC):
                raise RuntimeError(
                    "Release changed; review the data contract before downloading"
                )
    compressed_path = destination / "dataset23.member.part"
    downloaded = compressed_path.stat().st_size if compressed_path.exists() else 0
    if downloaded > MEMBER_BYTES:
        raise ValueError("Invalid partial-download size")
    if downloaded < MEMBER_BYTES:
        with request_range(OFFSET + downloaded, OFFSET + MEMBER_BYTES - 1) as response:
            with compressed_path.open("ab") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
    if compressed_path.stat().st_size != MEMBER_BYTES:
        raise RuntimeError("Incomplete download; run this command again to resume")
    temporary = csv_path.with_suffix(".csv.part")
    checksum = 0
    length = 0
    digest = hashlib.sha256()
    with compressed_path.open("rb") as source, temporary.open("wb") as output:
        header = struct.unpack("<4s5H3I2H", source.read(30))
        if header[0] != b"PK\x03\x04" or header[3] != 8:
            raise ValueError("Expected a deflated ZIP member")
        name = source.read(header[-2]).decode()
        if name != MEMBER_NAME:
            raise ValueError("Unexpected archive member name")
        source.read(header[-1])
        decompressor = zlib.decompressobj(-15)
        remaining = COMPRESSED_BYTES
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Truncated compressed member")
            remaining -= len(block)
            plain = decompressor.decompress(block)
            output.write(plain)
            digest.update(plain)
            checksum = zlib.crc32(plain, checksum)
            length += len(plain)
        if (
            not decompressor.eof
            or length != UNCOMPRESSED_BYTES
            or checksum != EXPECTED_CRC
        ):
            raise ValueError(
                "CSV length/CRC verification failed; do not use these bytes"
            )
    temporary.replace(csv_path)
    metadata = {
        "url": URL,
        "release": "7992926",
        "member": MEMBER_NAME,
        "archive_size": ARCHIVE_SIZE,
        "byte_range": [OFFSET, OFFSET + MEMBER_BYTES - 1],
        "archive_md5_published_not_verified_for_partial_fetch": "f7e8c553efee734cf161ee6f2b0a09cf",
        "member_crc32_verified": f"{checksum:08x}",
        "csv_bytes": length,
        "csv_sha256": digest.hexdigest(),
        "central_directory_tail_sha256": hashlib.sha256(tail).hexdigest(),
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw"
    )
    fetch(parser.parse_args().output)
