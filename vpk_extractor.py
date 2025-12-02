#!/usr/bin/env python3
import os
import struct
import sys


class VPKFormatError(Exception):
    pass


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected end of file (wanted {n} bytes, got {len(data)})")
    return data


def read_cstring(f):
    '''Read a null-terminated ASCII/UTF-8 string.'''
    chunks = []
    while True:
        b = f.read(1)
        if b == b"":
            raise EOFError("Unexpected EOF while reading C-string")
        if b == b"\x00":
            return b"".join(chunks).decode("utf-8", errors="replace")
        chunks.append(b)


def read_u16(f):
    return struct.unpack("<H", read_exact(f, 2))[0]


def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def parse_vpk_header(f):
    # Signature + version + tree size, then optional v2 fields.
    signature = read_u32(f)
    version = read_u32(f)
    if signature != 0x55AA1234:
        raise VPKFormatError("Not a VPK directory file (bad signature)")
    tree_size = read_u32(f)
    header_size = 12
    file_data_section_size = 0
    archive_md5_section_size = 0
    other_md5_section_size = 0
    signature_section_size = 0

    if version == 2:
        file_data_section_size = read_u32(f)
        archive_md5_section_size = read_u32(f)
        other_md5_section_size = read_u32(f)
        signature_section_size = read_u32(f)
        header_size = 28
    elif version != 1:
        raise VPKFormatError(f"Unsupported VPK version: {version}")

    return {
        "signature": signature,
        "version": version,
        "tree_size": tree_size,
        "header_size": header_size,
        "file_data_section_size": file_data_section_size,
        "archive_md5_section_size": archive_md5_section_size,
        "other_md5_section_size": other_md5_section_size,
        "signature_section_size": signature_section_size,
    }


def parse_vpk_tree(f):
    '''
    Parse the 3-level directory tree of a VPK.
    Returns a list of file entries.
    '''
    entries = []

    while True:
        ext = read_cstring(f)
        if ext == "":
            break
        if ext == " ":
            ext = ""

        while True:
            folder = read_cstring(f)
            if folder == "":
                break
            if folder == " ":
                folder = ""

            while True:
                filename = read_cstring(f)
                if filename == "":
                    break
                if filename == " ":
                    filename = ""

                crc = read_u32(f)
                preload_bytes = read_u16(f)
                archive_index = read_u16(f)
                entry_offset = read_u32(f)
                entry_length = read_u32(f)
                terminator = read_u16(f)
                if terminator != 0xFFFF:
                    raise VPKFormatError("Bad directory entry terminator")

                preload_data = b""
                if preload_bytes:
                    preload_data = read_exact(f, preload_bytes)

                # Build full path "folder/filename.ext"
                if folder:
                    base_path = folder.replace("\\", "/")
                else:
                    base_path = ""

                name = filename
                if ext:
                    if name:
                        name = f"{name}.{ext}"
                    else:
                        name = ext

                if base_path:
                    full_path = f"{base_path}/{name}"
                else:
                    full_path = name

                entries.append(
                    {
                        "path": full_path,
                        "crc": crc,
                        "preload_bytes": preload_bytes,
                        "archive_index": archive_index,
                        "entry_offset": entry_offset,
                        "entry_length": entry_length,
                        "preload_data": preload_data,
                    }
                )
    return entries


def extract_vpk(dir_vpk_path):
    dir_vpk_path = os.path.abspath(dir_vpk_path)
    dir_folder = os.path.dirname(dir_vpk_path)
    base_name = os.path.basename(dir_vpk_path)

    base_no_ext, _ = os.path.splitext(base_name)
    if base_no_ext.endswith("_dir"):
        # pak01_dir.vpk -> prefix "pak01", output folder "pak01"
        prefix = base_no_ext[:-4]
        out_dir_name = prefix
    else:
        # Single-file VPK or odd naming
        prefix = base_no_ext
        out_dir_name = base_no_ext

    out_dir = os.path.join(dir_folder, out_dir_name)

    print("=" * 60)
    print(f"Processing: {dir_vpk_path}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    # Open directory VPK
    with open(dir_vpk_path, "rb") as dir_f:
        header = parse_vpk_header(dir_f)
        print(f"VPK version: {header['version']}")
        print(f"Tree size : {header['tree_size']} bytes")
        print()

        # Tree starts immediately after header
        tree_start = dir_f.tell()
        entries = parse_vpk_tree(dir_f)
        tree_end = dir_f.tell()
        parsed_tree_size = tree_end - tree_start

        if parsed_tree_size != header["tree_size"]:
            print(
                f"Warning: header TreeSize={header['tree_size']} "
                f"but parsed {parsed_tree_size} bytes."
            )

        # Data for files stored in the dir file starts after header + tree
        dir_data_offset = header["header_size"] + header["tree_size"]

        print(f"Files in VPK: {len(entries)}")
        print()

        os.makedirs(out_dir, exist_ok=True)

        # Cache for archive file handles (pak01_000.vpk, pak01_001.vpk, etc.)
        archive_files = {}

        def get_archive_file(index):
            # 0x7FFF = data in the directory file itself
            if index == 0x7FFF:
                return dir_f
            if index in archive_files:
                return archive_files[index]

            archive_name = f"{prefix}_{index:03}.vpk"
            archive_path = os.path.join(dir_folder, archive_name)
            if not os.path.isfile(archive_path):
                raise FileNotFoundError(
                    f"Missing archive file: {archive_name} (needed for index {index})"
                )
            archive_files[index] = open(archive_path, "rb")
            return archive_files[index]

        for entry in entries:
            rel_path = entry["path"]
            if not rel_path:
                continue

            total_size = len(entry["preload_data"]) + entry["entry_length"]
            print(f"- {rel_path} ({total_size} bytes)")

            out_path = os.path.join(out_dir, *rel_path.split("/"))
            out_folder = os.path.dirname(out_path)
            if out_folder and not os.path.isdir(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            with open(out_path, "wb") as out_f:
                # Write preload bytes (if any)
                if entry["preload_data"]:
                    out_f.write(entry["preload_data"])

                # Write remaining data from archive/dir file
                if entry["entry_length"]:
                    src = get_archive_file(entry["archive_index"])
                    if entry["archive_index"] == 0x7FFF:
                        src.seek(dir_data_offset + entry["entry_offset"])
                    else:
                        src.seek(entry["entry_offset"])

                    remaining = entry["entry_length"]
                    chunk_size = 1024 * 1024
                    while remaining > 0:
                        to_read = min(chunk_size, remaining)
                        chunk = src.read(to_read)
                        if not chunk:
                            raise EOFError(
                                f"Unexpected EOF while reading data for '{rel_path}' "
                                f"(missing {remaining} bytes)"
                            )
                        out_f.write(chunk)
                        remaining -= len(chunk)

        # Close any opened archive files
        for f in archive_files.values():
            try:
                f.close()
            except Exception:
                pass

    print(f"Done with: {dir_vpk_path}")
    print()


def main():
    args = sys.argv[1:]

    if not args:
        print("Valve VPK extractor (drag & drop version)")
        print()
        print("Usage:")
        print("  - Drag one or more VPK directory files onto this script, e.g.:")
        print("      pak01_dir.vpk, tf2_textures_dir.vpk, etc.")
        print("  - For single-file VPKs, drag that .vpk directly.")
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

        if not path.lower().endswith(".vpk"):
            print("=" * 60)
            print(f"Warning: {path}")
            print("  (Doesn't look like a .vpk file, attempting anyway)")
            print()

        try:
            extract_vpk(path)
        except VPKFormatError as e:
            print(f"Error processing '{path}': {e}")
            print("If this is part of a multi-file VPK, drag the '*_dir.vpk' file instead.")
            print()
        except (EOFError, OSError, FileNotFoundError) as e:
            print(f"Error processing '{path}': {e}")
            print()

    input("All done. Press Enter to exit...")


if __name__ == "__main__":
    main()
