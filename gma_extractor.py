#!/usr/bin/env python3
import os
import struct
import sys


class GMAFormatError(Exception):
    pass


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected end of file (wanted {n} bytes, got {len(data)})")
    return data


def read_cstring(f):
    """Read a null-terminated UTF-8 string."""
    chunks = []
    while True:
        b = f.read(1)
        if b == b"":
            raise EOFError("Unexpected EOF while reading C-string")
        if b == b"\x00":
            return b"".join(chunks).decode("utf-8", errors="replace")
        chunks.append(b)


def read_u8(f):
    return struct.unpack("<B", read_exact(f, 1))[0]


def read_i8(f):
    return struct.unpack("<b", read_exact(f, 1))[0]


def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_i32(f):
    return struct.unpack("<i", read_exact(f, 4))[0]


def read_i64(f):
    return struct.unpack("<q", read_exact(f, 8))[0]


def read_u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]


def parse_gma_header(f):
    magic = read_exact(f, 4)
    if magic != b"GMAD":
        raise GMAFormatError("Not a GMA file (missing 'GMAD' magic)")

    version = read_i8(f)          # usually 3
    steam_id64 = read_i64(f)      # ignored
    timestamp = read_u64(f)       # ignored
    required_content = read_u8(f) # ignored

    addon_name = read_cstring(f)
    addon_desc = read_cstring(f)
    addon_author = read_cstring(f)
    addon_version = read_i32(f)

    return {
        "version": version,
        "steam_id64": steam_id64,
        "timestamp": timestamp,
        "required_content": required_content,
        "name": addon_name,
        "description": addon_desc,
        "author": addon_author,
        "addon_version": addon_version,
    }


def parse_gma_file_entries(f):
    """
    After the header, the format is:

      repeated:
        idx   (u32)  -- 1-based index, 0 means end of list
        name  (C-string)
        size  (i64)
        crc32 (u32)

    Then raw file data for each entry in the same order.
    """
    entries = []
    while True:
        idx = read_u32(f)
        if idx == 0:
            break

        name = read_cstring(f)
        size = read_i64(f)
        crc = read_u32(f)

        if size < 0:
            raise GMAFormatError(f"Negative file size for '{name}'")

        entries.append(
            {
                "index": idx,
                "name": name,
                "size": size,
                "crc32": crc,
            }
        )

    entries.sort(key=lambda e: e["index"])
    return entries


def extract_files(f, entries, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for entry in entries:
        rel_path = entry["name"].replace("\\", "/")
        parts = [p for p in rel_path.split("/") if p]
        out_path = os.path.join(out_dir, *parts)

        if parts:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        remaining = entry["size"]
        print(f"- Extracting {rel_path} ({remaining} bytes)")

        with open(out_path, "wb") as out_f:
            chunk_size = 1024 * 1024
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = f.read(to_read)
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF while reading data for '{rel_path}' "
                        f"(missing {remaining} bytes)"
                    )
                out_f.write(chunk)
                remaining -= len(chunk)


def extract_gma(gma_path):
    gma_path = os.path.abspath(gma_path)
    base = os.path.splitext(os.path.basename(gma_path))[0]
    out_dir = os.path.join(os.path.dirname(gma_path), base)

    print("=" * 60)
    print(f"Processing: {gma_path}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    with open(gma_path, "rb") as f:
        header = parse_gma_header(f)
        print(f"Addon name:     {header['name']}")
        print(f"Author:         {header['author']}")
        print(f"Addon version:  {header['addon_version']}")
        print(f"GMA fmt ver:    {header['version']}")
        print()

        entries = parse_gma_file_entries(f)
        print(f"Files in addon: {len(entries)}")
        print()

        extract_files(f, entries, out_dir)

    print(f"Done with: {gma_path}")
    print()


def main():
    args = sys.argv[1:]

    if not args:
        print("Garry's Mod .gma extractor (drag & drop version)")
        print()
        print("Usage:")
        print("  - Drag one or more .gma files onto this script, OR")
        print("  - Run from cmd:  python extract_gma_dragdrop.py addon.gma")
        print()
        input("No files provided. Press Enter to exit...")
        return

    for path in args:
        if not os.path.isfile(path):
            print("=" * 60)
            print(f"Skipping: {path}")
            print("  (Not a file or does not exist)")
            print()
            continue

        if not path.lower().endswith(".gma"):
            print("=" * 60)
            print(f"Warning: {path}")
            print("  (Doesn't look like a .gma file, attempting anyway)")
            print()

        try:
            extract_gma(path)
        except (GMAFormatError, EOFError, OSError) as e:
            print(f"Error processing '{path}': {e}")
            print()

    input("All done. Press Enter to exit...")


if __name__ == "__main__":
    main()
