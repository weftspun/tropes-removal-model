#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
"""Tiny streaming zstd (de)compressor -- standard .zst, interoperable with
the zstd CLI. Used to ship the parquet datalake and trained models as
Release assets and to inflate them on GHA runners.

    python zst.py -c data/ data.tar.zst   # tar+compress a directory
    python zst.py -c file file.zst        # compress a single file
    python zst.py -d file.zst file        # decompress
"""
import sys
import tarfile
import os
import zstandard


def compress(src, dst, level=19):
    if os.path.isdir(src):
        import io
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(src, arcname=os.path.basename(src))
        buf.seek(0)
        c = zstandard.ZstdCompressor(level=level, threads=-1)
        with open(dst, "wb") as o:
            c.copy_stream(buf, o)
    else:
        c = zstandard.ZstdCompressor(level=level, threads=-1)
        with open(src, "rb") as i, open(dst, "wb") as o:
            c.copy_stream(i, o)


def decompress(src, dst):
    d = zstandard.ZstdDecompressor()
    with open(src, "rb") as i, open(dst, "wb") as o:
        d.copy_stream(i, o)


if __name__ == "__main__":
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    (compress if mode in ("-c", "c", "compress") else decompress)(src, dst)
    print(f"{'compressed' if mode in ('-c', 'c', 'compress') else 'decompressed'} {src} -> {dst}")
