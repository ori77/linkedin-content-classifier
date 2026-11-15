"""Mutation check for the Step 17 exports. Break one line at a time; the tests must notice.
Run:  python tools/mutation_check_step17.py        (exit 0 = all killed in all three parts)

Targets:
  pipeline/step17_export.py     the export logic              tests/test_step17_export.py
  pipeline/dashboard_data.py    the three new review fields   tests/test_step17_export.py
  pipeline/step16_dashboard.py  the download buttons          tests/test_step16_dashboard.py

Every value in the two record builders (`library_records`, `correction_records`) is broken one by one
(replaced by None), so a column that no test looks at shows up as a survivor. The patterns for those
are made from the source itself, with enough neighbouring text to match exactly once.

Not listed on purpose (no test could tell the difference):
  * `io.StringIO(newline="")` against `io.StringIO()` (the csv writer adds the line ends itself);
  * the columns of `st.columns([1, 1, 2])` (layout only).
"""
import re
import sys
from pathlib import Path

from mutation_runner import ROOT, run_mutations

EXPORT = "pipeline/step17_export.py"
DATA = "pipeline/dashboard_data.py"
SCREEN = "pipeline/step16_dashboard.py"


def field_mutations(source: str) -> list[tuple[str, str, str]]:
    """Break each `"column": value,` of the two record builders, using as much neighbouring text as needed."""
    out = []
    for func in ("library_records", "correction_records"):
        start = source.index(f"def {func}(")
        body = source[start: source.index("\n\n\n", start)]
        pairs = list(re.finditer(r'"(\w+)": ([^,\n]+(?:\([^)\n]*\))?[^,\n]*),', body))
        for i, m in enumerate(pairs):
            old = m.group(0)
            context = old
            k = i
            while source.count(context) != 1 and k > 0:       # grow to the left until the text is unique
                k -= 1
                context = body[pairs[k].start(): m.end()]
            if source.count(context) != 1:
                continue
            new = context.replace(old, f'"{m.group(1)}": None,')
            out.append((f"{func}: {m.group(1)} lost", context, new))
    return out


EXPORT_MUTATIONS = [
    # --- one cell of CSV
    ("cell: None not empty", 'if value is None:\n        return ""', 'if value is None:\n        return "None"'),
    ("cell: true written as True", 'return "true" if value else "false"', 'return "True" if value else "False"'),
    ("cell: bool not handled", "    if isinstance(value, bool):", "    if False:"),
    ("cell: numbers can be defused", "    if isinstance(value, (int, float)):", "    if False:"),
    ("cell: list separator", 'LIST_SEPARATOR = " | "', 'LIST_SEPARATOR = "; "'),
    ("cell: lists not joined", "text = LIST_SEPARATOR.join(str(v) for v in value)", "text = str(value)"),
    ("cell: = not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("+", "-", "@", "\\t", "\\r")'),
    ("cell: + not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("=", "-", "@", "\\t", "\\r")'),
    ("cell: - not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("=", "+", "@", "\\t", "\\r")'),
    ("cell: @ not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("=", "+", "-", "\\t", "\\r")'),
    ("cell: tab not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("=", "+", "-", "@", "\\r")'),
    ("cell: return not defused", 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t", "\\r")', 'FORMULA_STARTS = ("=", "+", "-", "@", "\\t")'),
    ("cell: nothing defused", "    if text.startswith(FORMULA_STARTS):", "    if False:"),
    ("cell: everything defused", "    if text.startswith(FORMULA_STARTS):", "    if True:"),
    ("cell: defused without the quote", "        text = \"'\" + text", "        text = text"),
    # --- the two file formats
    ("csv: no byte-order mark", '.encode("utf-8-sig")', '.encode("utf-8")'),
    ("csv: header missing", "    writer.writerow(columns)\n", "    pass\n"),
    ("csv: rows not written", "        writer.writerow([csv_cell(rec[c]) for c in columns])", "        pass"),
    ("csv: cells not converted", "writer.writerow([csv_cell(rec[c]) for c in columns])", "writer.writerow([rec[c] for c in columns])"),
    ("json: unicode escaped", "json.dumps(doc, ensure_ascii=False, indent=2)", "json.dumps(doc, indent=2)"),
    ("json: not indented", "json.dumps(doc, ensure_ascii=False, indent=2)", "json.dumps(doc, ensure_ascii=False)"),
    ("json: no final line break", '+ "\\n").encode("utf-8")', ').encode("utf-8")'),
    ("json: wrong kind", '"kind": kind,', '"kind": "library",'),
    ("json: wrong time", '"exported_at": exported_at,', '"exported_at": "",'),
    ("json: wrong count", '"count": len(records),', '"count": 0,'),
    ("json: rows missing", '"rows": list(records)}', '"rows": []}'),
    # --- the records
    ("links: post link lost", '"post_url": row["post_url"],', '"post_url": None,'),
    ("links: first resource link lost", '"resource_link": row["resource_link"],', '"resource_link": None,'),
    ("links: resource links lost", '"resource_links": list(row["resource_links_list"]),', '"resource_links": [],'),
    ("links: resource link count lost", '"resource_link_count": row["resource_link_count"]}', '"resource_link_count": 0}'),
    ("corrections: only fixed posts", 'sorted((r for r in rows if r["reviewed"]), key=', 'sorted((r for r in rows if r["corrected_fields"]), key='),
    ("corrections: every post", 'sorted((r for r in rows if r["reviewed"]), key=', "sorted((r for r in rows), key="),
    ("corrections: decision swapped", '"corrected" if r["corrected_fields"] else "approved"', '"approved" if r["corrected_fields"] else "corrected"'),
    ("corrections: newest review first", 'key=lambda r: (r["reviewed_at"], r["activity_id"])', 'key=lambda r: (r["activity_id"], r["reviewed_at"])'),
    ("corrections: ties not broken by post id", 'key=lambda r: (r["reviewed_at"], r["activity_id"])', 'key=lambda r: (r["reviewed_at"],)'),
    ("corrections: oldest review last", 'key=lambda r: (r["reviewed_at"], r["activity_id"])', 'key=lambda r: (r["reviewed_at"], -r["activity_id"])'),
    ("library: id is a number", '"activity_id": str(r["activity_id"]), "title": r["title"], "author": r["author"],\n            "saved_at"', '"activity_id": r["activity_id"], "title": r["title"], "author": r["author"],\n            "saved_at"'),
    ("corrections: id is a number", '"activity_id": str(r["activity_id"]), "title": r["title"], "author": r["author"],\n            **_links(r),\n            "review_id"', '"activity_id": r["activity_id"], "title": r["title"], "author": r["author"],\n            **_links(r),\n            "review_id"'),
    ("library: title lost", '"title": r["title"], "author": r["author"],\n            "saved_at"', '"title": None, "author": r["author"],\n            "saved_at"'),
    ("library: author lost", '"title": r["title"], "author": r["author"],\n            "saved_at"', '"title": r["title"], "author": None,\n            "saved_at"'),
    ("corrections: title lost", '"title": r["title"], "author": r["author"],\n            **_links(r),\n            "review_id"', '"title": None, "author": r["author"],\n            **_links(r),\n            "review_id"'),
    ("corrections: author lost", '"title": r["title"], "author": r["author"],\n            **_links(r),\n            "review_id"', '"title": r["title"], "author": None,\n            **_links(r),\n            "review_id"'),
    ("library: links not in the column list", '"post_url", "resource_link", "resource_links", "resource_link_count",\n    "topic_id"', '"topic_id"'),
    ("corrections: links not in the column list", '"post_url", "resource_link", "resource_links", "resource_link_count",\n    "review_id"', '"review_id"'),
    # --- writing files
    ("write: file written in place", "        os.replace(tmp, path)", "        path.write_bytes(data)"),
    ("write: temporary file kept after a failure", "        if tmp.exists():\n            tmp.unlink()", "        pass"),
    ("write: temporary file never written", "        tmp.write_bytes(data)\n", "        pass\n"),
    ("export: any --what accepted", "    if what not in WHATS:", "    if False:"),
    ("export: any --format accepted", "    if fmt not in FORMATS:", "    if False:"),
    ("export: time ignored", "stamp = exported_at or D.utc_now()\n    out_dir", "stamp = D.utc_now()\n    out_dir"),
    ("export: no default time", "stamp = exported_at or D.utc_now()\n    out_dir", "stamp = exported_at\n    out_dir"),
    ("export: no parent folders", "out_dir.mkdir(parents=True, exist_ok=True)", "out_dir.mkdir(exist_ok=True)"),
    ("export: existing folder refused", "out_dir.mkdir(parents=True, exist_ok=True)", "out_dir.mkdir(parents=True)"),
    ("export: folder error not explained", 'except OSError as exc:\n        raise ExportError(f"cannot create', 'except ZeroDivisionError as exc:\n        raise ExportError(f"cannot create'),
    ("export: write error not explained", 'except OSError as exc:\n            raise ExportError(f"cannot write', 'except ZeroDivisionError as exc:\n            raise ExportError(f"cannot write'),
    ("export: library always", '    if what in ("library", "both"):', "    if True:"),
    ("export: library never", '    if what in ("library", "both"):', "    if False:"),
    ("export: corrections always", '    if what in ("corrections", "both"):', "    if True:"),
    ("export: corrections never", '    if what in ("corrections", "both"):', "    if False:"),
    ("export: csv always", '            if fmt in ("csv", "both"):', "            if True:"),
    ("export: csv never", '            if fmt in ("csv", "both"):', "            if False:"),
    ("export: json always", '            if fmt in ("json", "both"):', "            if True:"),
    ("export: json never", '            if fmt in ("json", "both"):', "            if False:"),
    ("export: csv file name", 'path = out_dir / f"{kind}.csv"', 'path = out_dir / f"{kind}.txt"'),
    ("export: json file name", 'path = out_dir / f"{kind}.json"', 'path = out_dir / f"{kind}.txt"'),
    ("export: csv columns of the other kind", "plan.append((\"library\", library_records(rows), LIBRARY_COLUMNS))", "plan.append((\"library\", library_records(rows), CORRECTION_COLUMNS))"),
    ("export: wrong records for corrections", 'plan.append(("corrections", correction_records(rows), CORRECTION_COLUMNS))', 'plan.append(("corrections", library_records(rows), CORRECTION_COLUMNS))'),
    ("export: row counts wrong", "written.append((path, len(records)))\n            if fmt in", "written.append((path, 0))\n            if fmt in"),
    ("export: json row counts wrong", "                write_atomic(path, to_json_bytes(records, kind, stamp))\n                written.append((path, len(records)))", "                write_atomic(path, to_json_bytes(records, kind, stamp))\n                written.append((path, 0))"),
    # --- the download buttons
    ("buttons: csv key", 'f"{key_prefix}_dl_csv"', 'f"{key_prefix}_csv"'),
    ("buttons: json key", 'f"{key_prefix}_dl_json"', 'f"{key_prefix}_json"'),
    ("buttons: csv label", 'f"⬇ {name} as CSV"', 'f"⬇ {name} as csv file"'),
    ("buttons: json label", 'f"⬇ {name} as JSON"', 'f"⬇ {name} as json file"'),
    ("buttons: csv data", '"data": to_csv_bytes(records, columns),', '"data": b"",'),
    ("buttons: json data", '"data": to_json_bytes(records, kind, stamp),', '"data": b"",'),
    ("buttons: csv file name", '"file_name": f"{kind}.csv", "mime": "text/csv"', '"file_name": f"{kind}.txt", "mime": "text/csv"'),
    ("buttons: json file name", '"file_name": f"{kind}.json", "mime": "application/json"', '"file_name": f"{kind}.txt", "mime": "application/json"'),
    ("buttons: csv type", '"mime": "text/csv"', '"mime": "text/plain"'),
    ("buttons: json type", '"mime": "application/json"', '"mime": "text/plain"'),
    ("buttons: csv enabled when empty", '"mime": "text/csv", "disabled": n == 0}', '"mime": "text/csv", "disabled": False}'),
    ("buttons: json enabled when empty", '"mime": "application/json", "disabled": n == 0}', '"mime": "application/json", "disabled": False}'),
    ("buttons: disabled when not empty", '"mime": "text/csv", "disabled": n == 0}', '"mime": "text/csv", "disabled": n != 0}'),
    ("buttons: count in the label ignored", 'name = phrase.format(n=n)', 'name = phrase.format(n=0)'),
    ("buttons: time ignored", "stamp = exported_at or D.utc_now()\n    return [", "stamp = D.utc_now()\n    return ["),
    ("buttons: library file kind", '"library", LIBRARY_COLUMNS, "lib", "These {n} posts"', '"corrections", LIBRARY_COLUMNS, "lib", "These {n} posts"'),
    ("buttons: library columns", '"library", LIBRARY_COLUMNS, "lib", "These {n} posts"', '"library", CORRECTION_COLUMNS, "lib", "These {n} posts"'),
    ("buttons: library key prefix", '"library", LIBRARY_COLUMNS, "lib", "These {n} posts"', '"library", LIBRARY_COLUMNS, "rv", "These {n} posts"'),
    ("buttons: library wording", '"These {n} posts"', '"All {n} posts"'),
    ("buttons: corrections file kind", '"corrections", CORRECTION_COLUMNS, "rv", "My {n} reviews"', '"library", CORRECTION_COLUMNS, "rv", "My {n} reviews"'),
    ("buttons: corrections columns", '"corrections", CORRECTION_COLUMNS, "rv", "My {n} reviews"', '"corrections", LIBRARY_COLUMNS, "rv", "My {n} reviews"'),
    ("buttons: corrections key prefix", '"corrections", CORRECTION_COLUMNS, "rv", "My {n} reviews"', '"corrections", CORRECTION_COLUMNS, "lib", "My {n} reviews"'),
    ("buttons: corrections wording", '"My {n} reviews"', '"All {n} reviews"'),
    ("buttons: corrections use library records", "return _downloads(correction_records(rows),", "return _downloads(library_records(rows),"),
    # --- the command line
    ("cli: environment ignored", 'default=env.get("LISC_DB") or D.DEFAULT_DB', "default=D.DEFAULT_DB"),
    ("cli: empty environment value used", 'default=env.get("LISC_DB") or D.DEFAULT_DB', 'default=env.get("LISC_DB", D.DEFAULT_DB)'),
    ("cli: wrong default folder", 'DEFAULT_OUT_DIR = "exports"', 'DEFAULT_OUT_DIR = "out"'),
    ("cli: default what", 'choices=WHATS, default="both"', 'choices=WHATS, default="library"'),
    ("cli: default format", 'choices=FORMATS, default="both"', 'choices=FORMATS, default="csv"'),
    ("cli: errors exit 0", '        print(f"ERROR: {exc}", file=sys.stderr)\n        return 2', '        print(f"ERROR: {exc}", file=sys.stderr)\n        return 0'),
    ("cli: errors not printed", '        print(f"ERROR: {exc}", file=sys.stderr)\n', "        pass\n"),
    ("cli: database errors not caught", "except (D.DashboardError, ExportError) as exc:", "except ExportError as exc:"),
    ("cli: export errors not caught", "except (D.DashboardError, ExportError) as exc:", "except D.DashboardError as exc:"),
    ("cli: files not listed", '        print(f"wrote {path}  ({n} rows)")', "        pass"),
    ("cli: reviewed count wrong", 'reviewed = sum(1 for r in rows if r["reviewed"])', 'reviewed = sum(1 for r in rows if r["model_flagged"])'),
    ("cli: options ignored", "args = parser.parse_args(argv)", "args = parser.parse_args([])"),
    ("cli: format not passed", "export_files(rows, args.out_dir, args.what, args.fmt)", 'export_files(rows, args.out_dir, args.what, "csv")'),
    ("cli: what not passed", "export_files(rows, args.out_dir, args.what, args.fmt)", 'export_files(rows, args.out_dir, "both", args.fmt)'),
    ("cli: folder not passed", "export_files(rows, args.out_dir, args.what, args.fmt)", 'export_files(rows, DEFAULT_OUT_DIR, args.what, args.fmt)'),
]

DATA_MUTATIONS = [
    ("review id not read", 'row["review_id"] = review["id"]', 'row["review_id"] = None'),
    ("review time not read", 'row["reviewed_at"] = review["created_at"]', 'row["reviewed_at"] = None'),
    ("example flag not read", 'row["use_as_example"] = bool(review["use_as_example"])', 'row["use_as_example"] = True'),
    ("example flag not a boolean", 'row["use_as_example"] = bool(review["use_as_example"])', 'row["use_as_example"] = review["use_as_example"]'),
    ("review fields not empty without a review", 'row["review_id"] = row["reviewed_at"] = row["use_as_example"] = None', 'row["review_id"] = row["reviewed_at"] = row["use_as_example"] = ""'),
]

SCREEN_MUTATIONS = [
    ("screen: library buttons not shown", "        show_downloads(X.library_downloads(shown))\n", "        pass\n"),
    ("screen: library buttons for all posts", "show_downloads(X.library_downloads(shown))", "show_downloads(X.library_downloads(rows))"),
    ("screen: review buttons not shown", "    show_downloads(X.correction_downloads(rows))\n", "    pass\n"),
    ("screen: review buttons from the library list", "show_downloads(X.correction_downloads(rows))", "show_downloads(X.library_downloads(rows))"),
    ("screen: disabled ignored", 'disabled=spec["disabled"], key=spec["key"])', 'disabled=False, key=spec["key"])'),
    ("screen: key ignored", 'disabled=spec["disabled"], key=spec["key"])', 'disabled=spec["disabled"], key=spec["key"] + "x")'),
    ("screen: label ignored", 'column.download_button(spec["label"], spec["data"],', 'column.download_button("Download", spec["data"],'),
    ("screen: only one button", "for column, spec in zip(st.columns([1, 1, 2]), specs):", "for column, spec in zip(st.columns([1, 1, 2]), specs[:1]):"),
]


if __name__ == "__main__":
    export_source = (ROOT / EXPORT).read_text(encoding="utf-8")
    code = run_mutations(EXPORT, "tests/test_step17_export.py", field_mutations(export_source) + EXPORT_MUTATIONS)
    code |= run_mutations(DATA, "tests/test_step17_export.py", DATA_MUTATIONS)
    code |= run_mutations(SCREEN, "tests/test_step16_dashboard.py", SCREEN_MUTATIONS)
    sys.exit(code)
