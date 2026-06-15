"""Data IO helpers and path resolution for DNAscape."""

from pathlib import Path
from collections import defaultdict
import gzip
import re
import shutil

import numpy as np
import pandas as pd

from .constants import mapt

_MODULE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_ROOT.parent

def _resolve_data_path(relative_path, for_write=False):
    rel = Path(relative_path)
    if for_write:
        target = Path('data') / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    search_bases = (
        Path('data'),
        Path('examples/data'),
        _PROJECT_ROOT / 'data',
        _PROJECT_ROOT / 'examples' / 'data',
    )
    for base in search_bases:
        cand = base / rel
        if cand.exists():
            return cand
    return Path('data') / rel

for folder in (
    Path("figures"),
    Path("data/firing_rate"),
    Path("data/replication_timing"),
    Path("data/fork_directionality"),
    Path("data/efficiency"),
    Path("data/output"),
    Path("data/fit_error"),
    Path("data/region"),
    Path("data/other"),
):
    folder.mkdir(parents=True, exist_ok=True)

def load(d='replication_timing', cell_line='H1', chr_number=1, example='hESC'):
    if d != 'example':
        data = np.loadtxt(_resolve_data_path(f"{mapt[d]}/{mapt[d]}_{cell_line}_chr{chr_number}.txt"), dtype=float)
    else:
        data = np.loadtxt(_resolve_data_path(f"{mapt[d]}/{example}_chr{chr_number}.txt"), dtype=float)
    return data

def loadtxt(stitle='save_example', outpath="output"):
    arr = np.loadtxt(
        _resolve_data_path(f"{outpath}/{stitle}.txt"),
        dtype=float
    )
    return arr

def loadcsv(d='replication_timing', cell_line='H1', chr_number=1, example='hESC', region='region'):
    if d == 'example':
        fname = _resolve_data_path(f"{mapt[d]}/{example}.csv")
    elif d == 'region':
        fname = _resolve_data_path(f"{mapt[d]}/{region}.csv")
    else:
        fname = _resolve_data_path(f"{mapt[d]}/{mapt[d]}_{cell_line}.csv")
    df = pd.read_csv(fname)
    col = f"chr{chr_number}"
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in {fname}")
    data = df[col].dropna().to_numpy(dtype=float)
    return data

def savetxt(arr, stitle='save_example', outpath="output"):
    np.savetxt(
        _resolve_data_path(f"{outpath}/{stitle}.txt", for_write=True),
        np.asarray(arr, float)
    )

def savecsv(arr, chr_number, stitle='save_example', outpath="output"):
    fname = _resolve_data_path(f"{outpath}/{stitle}.csv", for_write=True)
    col = f"chr{chr_number}"
    arr = np.asarray(arr, float)
    df = pd.read_csv(fname) if fname.exists() else pd.DataFrame()
    if len(df) < len(arr):
        df = df.reindex(range(len(arr)))
    df[col] = pd.Series(arr, index=range(len(arr)))
    cols = list(df.columns)
    cols_sorted = sorted(
        cols,
        key=lambda c: int(c[3:]) if c.startswith("chr") else float("inf")
    )
    df = df[cols_sorted]
    df.to_csv(fname, index=False)

def csv_to_tsv(input_path, output_path=None, decimal_places=2, encoding="utf-8", input_sep=None):
    """Convert a separated-value table to a tab-separated table."""
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".tsv") if output_path is None else Path(output_path)

    if decimal_places is None:
        float_format = None
    else:
        decimal_places = int(decimal_places)
        if decimal_places < 0:
            raise ValueError("decimal_places must be non-negative or None.")
        float_format = f"%.{decimal_places}f"

    read_kwargs = {"encoding": encoding}
    if input_sep is None:
        read_kwargs.update({"sep": None, "engine": "python"})
    else:
        read_kwargs["sep"] = input_sep

    df = pd.read_csv(input_path, **read_kwargs)
    if decimal_places is not None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
    df.to_csv(output_path, sep="\t", index=False, float_format=float_format, lineterminator="\n")

    return output_path

def compress_to_gz(input_path, output_path=None, compresslevel=9):
    """Compress a file with gzip and return the written .gz path."""
    input_path = Path(input_path)
    output_path = Path(f"{input_path}.gz") if output_path is None else Path(output_path)

    with input_path.open("rb") as source, gzip.open(output_path, "wb", compresslevel=compresslevel) as target:
        shutil.copyfileobj(source, target)

    return output_path

def _parse_gtf_attributes(attr_str: str) -> dict:
    out = {}
    for m in re.finditer(r'(\S+)\s+"([^"]*)"\s*;', attr_str):
        out[m.group(1)] = m.group(2)
    return out

def load_gene_models_from_gtf_gz(
    gtf_gz_path: str,
    chrom: str,
    start: int,
    end: int,
    keep_gene_types=None,
    max_genes=200
):
    genes = {}
    exons_by_gene = defaultdict(list)

    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqname, source, feature, s, e, score, strand, frame, attrs = fields
            if seqname != chrom:
                continue

            s = int(s) - 1
            e = int(e)
            if e <= start or s >= end:
                continue

            ad = _parse_gtf_attributes(attrs)
            gene_id = ad.get("gene_id")
            gene_name = ad.get("gene_name", gene_id)

            if keep_gene_types is not None:
                gene_type = ad.get("gene_type") or ad.get("gene_biotype")
                if gene_type not in keep_gene_types:
                    continue

            if feature == "gene":
                if gene_id is None:
                    continue
                genes[gene_id] = {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "chrom": seqname,
                    "start": s,
                    "end": e,
                    "strand": strand,
                }

            elif feature == "exon":
                if gene_id is None:
                    continue
                exons_by_gene[gene_id].append((s, e))

            if len(genes) > max_genes:
                break

    for gid, exs in exons_by_gene.items():
        if gid not in genes:
            min_s = min(x[0] for x in exs)
            max_e = max(x[1] for x in exs)
            genes[gid] = {
                "gene_id": gid,
                "gene_name": gid,
                "chrom": chrom,
                "start": min_s,
                "end": max_e,
                "strand": ".",
            }
        genes[gid]["exons"] = sorted(exs)

    out = list(genes.values())
    out.sort(key=lambda g: (g["start"], g["end"]))
    return out
