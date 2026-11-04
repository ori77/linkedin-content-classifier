"""
merge_export.py: join the LinkedIn export CSV with the parsed saved-page posts.

Inputs
------
1. EXPORT.csv   LinkedIn data export file ("Saved Items.csv"), columns:
                   savedItem    https://www.linkedin.com/feed/update/urn:li:activity:<ID>
                   CreatedTime  save time, DAY-FIRST, e.g. "30-10-2026 13:50"
2. PARSED.csv   output of parse_saved.py

How the join works (all offline, nothing is fetched)
----------------------------------------------------
* The activity ID is cut out of each export URL as plain text.
* Each parsed post already carries the same ID (from its card tag).
* The two lists are matched on that number. Every row from BOTH sides is kept.

status column
-------------
captured             in export and in captures  -> ready to classify
not_yet_captured     in export only             -> scroll further / maybe deleted
saved_after_export   in captures only           -> saved after you requested the export

Output: MERGED.csv = parsed columns + saved_at + status, plus a short coverage
report printed to the screen.

Usage
-----
python merge_export.py EXPORT.csv PARSED.csv MERGED.csv
"""
from __future__ import annotations

import re
import sys

import pandas as pd

ID_RE = re.compile(r"urn:li:activity:(\d+)")


def load_export(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    url_col = next(c for c in df.columns if c.strip().lower() == "saveditem")
    time_col = next(c for c in df.columns if c.strip().lower() == "createdtime")
    df["activity_id"] = df[url_col].str.extract(ID_RE, expand=False)
    bad = df["activity_id"].isna().sum()
    if bad:
        print(f"  ! {bad} export rows have no activity ID (not a post URL); kept aside")
    df = df.dropna(subset=["activity_id"]).copy()
    df["activity_id"] = df["activity_id"].astype("int64")
    df["saved_at"] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    df = (df.sort_values("saved_at", ascending=False)
            .drop_duplicates("activity_id")[["activity_id", "saved_at"]])
    return df


def main(export_csv: str, parsed_csv: str, out_csv: str) -> None:
    exp = load_export(export_csv)
    posts = pd.read_csv(parsed_csv, encoding="utf-8-sig")
    posts["activity_id"] = posts["activity_id"].astype("int64")

    m = exp.merge(posts, on="activity_id", how="outer", indicator=True)
    m["status"] = m["_merge"].map({
        "both": "captured",
        "left_only": "not_yet_captured",
        "right_only": "saved_after_export",
    })
    m = m.drop(columns="_merge").sort_values("saved_at", ascending=False, na_position="first")
    m.to_csv(out_csv, index=False, encoding="utf-8-sig")

    n_exp = len(exp)
    n_cap = (m.status == "captured").sum()
    print(f"export rows (unique posts) : {n_exp}")
    print(f"captured                   : {n_cap}  ({n_cap / max(n_exp, 1):.0%})")
    print(f"not yet captured           : {(m.status == 'not_yet_captured').sum()}")
    print(f"saved after export         : {(m.status == 'saved_after_export').sum()}")
    cap = m[m.status == "captured"]
    if len(cap):
        print(f"captured saves span        : {cap.saved_at.min():%d-%m-%Y} -> {cap.saved_at.max():%d-%m-%Y}")
    print(f"wrote {len(m)} rows -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
