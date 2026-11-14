"""Mutation check for the Step 15 loader. Break one line at a time; the tests must notice.
Run:  python tools/mutation_check_step15.py        (exit 0 = all killed)

Target: pipeline/step15_load_sqlite.py, tests: tests/test_step15_load_sqlite.py.

Not listed on purpose (no test could ever tell the difference):
  * `.strip()` on the post link when it is written: the link was already checked and the CSV
    reader returns the same text, so only the stray-spaces test sees it (it is listed below).
  * the `ON DELETE CASCADE` words on the tables other than items (the one test deletes an item).
"""
import sys

from mutation_runner import run_mutations

T = "pipeline/step15_load_sqlite.py"
TESTS = "tests/test_step15_load_sqlite.py"

MUTATIONS = [
    # --- the review rule re-applied
    ("rescore: clustering noise counts as disagreement", "if clustering == I.OTHER_ID:", "if clustering is None:"),
    ("rescore: every post counts as first try", 'first_try = parts["valid"] >= C.CONFIDENCE["valid_first_try"]', "first_try = True"),
    ("rescore: partial text ignored", 'partial=bool(parts["partial"])', "partial=False"),
    ("rescore: warning penalties ignored", 'penalties=int(parts["penalties"])', "penalties=0"),
    ("rescore: the model's own confidence ignored", 'self_confidence=parts["self"]', "self_confidence=1.0"),
    ("rescore: the rule is not applied", "reasons = RV.review_reasons(rule,", "reasons = [] and RV.review_reasons(rule,"),
    ("rescore: needs_review always true", '"needs_review": bool(reasons)}', '"needs_review": True}'),
    ("rescore: reasons not returned", '"review_reasons": reasons,', '"review_reasons": [],'),
    # --- links and titles
    ("links: empty pieces kept", "if u.strip()]", "]"),
    ("links: spaces kept", "[u.strip() for u in", "[u for u in"),
    ("title: attachment title ignored", "if attachment_title:\n        return", "if False:\n        return"),
    ("title: attachment title not cut", "return attachment_title[:TITLE_CHARS]", "return attachment_title"),
    ("title: blank first line used", "        if line:\n            return line if", "        if True:\n            return line if"),
    ("title: line exactly at the limit cut", "return line if len(line) <= TITLE_CHARS else", "return line if len(line) < TITLE_CHARS else"),
    ("title: cut line too long", "line[:TITLE_CHARS - 1].rstrip()", "line[:TITLE_CHARS].rstrip()"),
    # --- the posts CSV
    ("csv: any text accepted as a link", 'if not url.startswith(("http://", "https://")):', "if not url:"),
    ("csv: missing link accepted", 'if not url.startswith(("http://", "https://")):', "if False:"),
    ("csv: repeated id accepted", "if aid in rows:", "if False:"),
    ("csv: bad id crashes", "except ValueError:\n                    problems.append(f\"line {line_no}: activity_id '{row", "except ZeroDivisionError:\n                    problems.append(f\"line {line_no}: activity_id '{row"),
    ("csv: missing columns not reported", "if missing:\n                raise LoadError(f\"posts CSV", "if False:\n                raise LoadError(f\"posts CSV"),
    ("csv: byte order mark not skipped", 'path.open(newline="", encoding="utf-8-sig")', 'path.open(newline="", encoding="utf-8")'),
    ("csv: unreadable file crashes", "except OSError as exc:\n        raise LoadError(f\"posts CSV", "except ValueError as exc:\n        raise LoadError(f\"posts CSV"),
    ("csv: problems not raised", 'if problems:\n        raise LoadError("posts CSV problems', 'if False:\n        raise LoadError("posts CSV problems'),
    # --- the results file
    ("results: failure lines not skipped", 'if "error" in rec and "subtopic_id" not in rec:', "if False:"),
    ("results: repeated lines not counted", "            duplicates += 1", "            duplicates += 0"),
    ("results: first line wins", 'by_id[rec["activity_id"]] = rec', 'by_id.setdefault(rec["activity_id"], rec)'),
    ("results: missing fields accepted", "        if missing:\n            problems.append(f\"line {line_no}: activity_id", "        if False:\n            problems.append(f\"line {line_no}: activity_id"),
    ("results: bad JSON crashes", "except json.JSONDecodeError as exc:", "except ZeroDivisionError as exc:"),
    ("results: no activity id accepted", 'if not isinstance(rec, dict) or not isinstance(rec.get("activity_id"), int):', "if not isinstance(rec, dict):"),
    ("results: blank lines parsed", "        if not line.strip():\n            continue", "        if False:\n            continue"),
    ("results: byte order mark not skipped", 'raw.decode("utf-8-sig")', 'raw.decode("utf-8")'),
    ("results: problems not raised", 'if problems:\n        raise LoadError("results problems:\\n- " + "\\n- ".join(problems))\n    return list', 'if False:\n        raise LoadError("results problems:\\n- " + "\\n- ".join(problems))\n    return list'),
    ("results: unreadable file crashes", "except OSError as exc:\n        raise LoadError(f\"results problems", "except ValueError as exc:\n        raise LoadError(f\"results problems"),
    ("results: empty file accepted", "    if not records:\n", "    if False:\n"),
    ("check: post missing from the CSV accepted", "        if aid not in posts:", "        if False:"),
    ("check: other taxonomy version accepted", 'if rec["taxonomy_version"] != taxonomy.version:', "if False:"),
    ("check: unknown subtopic accepted", 'if rec["subtopic_id"] not in valid:', "if False:"),
    ("check: list cut too late", "problems[:50]", "problems[:51]"),
    ("check: wrong 'more' count", "len(problems) - 50", "len(problems) - 49"),
    ("check: long list never cut", "if len(problems) > 50", "if len(problems) > 5000"),
    ("load: problems not raised", 'if problems:\n        raise LoadError("results problems:\\n- " + "\\n- ".join(problems))\n\n    report', 'if False:\n        raise LoadError("results problems:\\n- " + "\\n- ".join(problems))\n\n    report'),
    # --- the database file
    ("db: foreign keys off", 'con.execute("PRAGMA foreign_keys = ON")', 'con.execute("PRAGMA foreign_keys = OFF")'),
    ("db: foreign file accepted", "    elif version != SCHEMA_VERSION:", "    elif False:"),
    ("db: empty file not filled", "if new or (version == 0 and not con.execute", "if new or (False and not con.execute"),
    ("db: folder not created", "path.parent.mkdir(parents=True, exist_ok=True)", "pass"),
    ("db: version not stored", 'con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")', 'con.execute("PRAGMA user_version = 0")'),
    ("db: always rebuilt", "new = not path.exists()", "new = True"),
    # --- the taxonomy
    ("taxonomy: changed content accepted", 'if row["sha256"] != taxonomy.sha:', "if False:"),
    ("taxonomy: stored twice", "    if row is not None:\n        if row", "    if False:\n        if row"),
    ("taxonomy: wrong name for the other topic", 'topic_names[I.OTHER_TOPIC_ID] = other["name"].strip()', 'topic_names[I.OTHER_TOPIC_ID] = "x"'),
    ("taxonomy: wrong name for other.general", "rows.append((taxonomy.version, I.OTHER_ID, I.OTHER_TOPIC_ID, other[\"name\"].strip()))", 'rows.append((taxonomy.version, I.OTHER_ID, I.OTHER_TOPIC_ID, "x"))'),
    ("taxonomy: tree not stored", "(taxonomy.version, taxonomy.sha, _js(tree), now)", "(taxonomy.version, taxonomy.sha, _js({}), now)"),
    # --- items and contents
    ("item: last resource link used as the first", "links[0] if links else None", "links[-1] if links else None"),
    ("item: link count wrong", "_js(links), links[0] if links else None, len(links),", "_js(links), links[0] if links else None, 0,"),
    ("item: links not stored", "_js(links), links[0] if links else None", "'[]', links[0] if links else None"),
    ("item: textless post marked classified", "status = CLASSIFIED if has_result else NO_TEXT", "status = CLASSIFIED"),
    ("item: empty save date stored as text", '(row.get("saved_at") or "").strip() or None', '(row.get("saved_at") or "").strip()'),
    ("item: collector default lost", '(row.get("source_file") or "").strip() or UNKNOWN_COLLECTOR', '(row.get("source_file") or "").strip()'),
    ("item: capture status default lost", '(row.get("status") or "").strip() or "captured"', '(row.get("status") or "").strip()'),
    ("item: empty author stored as text", '(row.get("author") or "").strip() or None,', '(row.get("author") or "").strip(),'),
    ("item: post date not stored", '(row.get("post_date_utc") or "").strip() or None, ', "None, "),
    ("item: raw text not used", "(row.get(\"text_clean\") or \"\").strip() or (row.get(\"text\") or \"\").strip()", "(row.get(\"text_clean\") or \"\").strip()"),
    ("item: old content kept when the text goes", 'con.execute("DELETE FROM contents WHERE item_id = ?", (str(aid),))', "pass"),
    ("item: link stored with spaces", 'row["post_url"].strip(), _js(links)', 'row["post_url"], _js(links)'),
    ("item: link not updated when it changes", "post_url=excluded.post_url, resource_links", "resource_links"),
    ("content: image flag wrong", '.lower() == "true"', '.lower() == "false"'),
    ("content: link count missing", '"url_count": len(links),', '"url_count": 0,'),
    ("content: hash of something else", "hashlib.sha256(text.encode(\"utf-8\")).hexdigest()", "hashlib.sha256(b\"x\").hexdigest()"),
    ("content: partial flag lost", "_js(metrics), int(partial)))", "_js(metrics), 0))"),
    ("content: partial never detected", 'partial = bool(rec) and "text_partial" in rec["warnings"]', "partial = False"),
    ("content: text not updated", "ON CONFLICT(item_id) DO UPDATE SET body_text=excluded.body_text,", "ON CONFLICT(item_id) DO UPDATE SET"),
    # --- the run
    ("run: start time wrong", "        stamps[0], stamps[-1]))", "        stamps[-1], stamps[-1]))"),
    ("run: end time wrong", "        stamps[0], stamps[-1]))", "        stamps[0], stamps[0]))"),
    ("run: token sum lost", 'sum(r.get("input_tokens") or 0 for r in records)', "0"),
    ("run: not updated when the file grows", "items_ok=excluded.items_ok,", "items_ok=items_ok,"),
    ("run: file fingerprint missing", '"file_sha256": file_sha,', '"file_sha256": "",'),
    ("run: id not from the file name", '"run_" + jsonl_path.stem', '"run_x"'),
    # --- classifications
    ("write: older row stays active", "SET is_active = 0 WHERE item_id = ? AND is_active = 1", "SET is_active = 1 WHERE item_id = ? AND is_active = 1"),
    ("write: older rows not counted", "report.superseded += switched", "report.superseded += 0"),
    ("write: known results inserted again", "                if existing:\n", "                if False:\n"),
    ("write: refresh leaves the old confidence", "(new[\"confidence\"], _js(new[\"confidence_parts\"]), int(new[\"needs_review\"]),\n                         _js(new[\"review_reasons\"]), rule.version, rule.sha, existing[\"id\"])", "(0, '{}', 0,\n                         '[]', rule.version, rule.sha, existing[\"id\"])"),
    ("write: refresh not counted", "report.refreshed += 1", "report.refreshed += 0"),
    ("write: new not counted", "report.inserted += 1", "report.inserted += 0"),
    ("write: run values lost", "rec[\"confidence\"], int(bool(rec[\"needs_review\"])),", "0, 0,"),
    ("write: recomputed confidence lost", "        new[\"confidence\"], _js(new[\"confidence_parts\"]), int(new[\"needs_review\"]), _js(new[\"review_reasons\"]),\n        new[\"rule_version\"]", "        rec[\"confidence\"], _js(new[\"confidence_parts\"]), int(new[\"needs_review\"]), _js(new[\"review_reasons\"]),\n        new[\"rule_version\"]"),
    ("write: needs_review taken from the run", "int(new[\"needs_review\"]), _js(new[\"review_reasons\"]),\n        new[\"rule_version\"]", "int(bool(rec[\"needs_review\"])), _js(new[\"review_reasons\"]),\n        new[\"rule_version\"]"),
    ("write: classification id not from the cache key", '"cls_" + rec["cache_key"][:16]', '"cls_" + str(rec["activity_id"])'),
    ("write: time basis lost", 't["cover_bucket"], t["understand_bucket"], t["basis"], _js(t),', 't["cover_bucket"], t["understand_bucket"], "x", _js(t),'),
    ("write: understand time taken from cover time", 'rec["difficulty"], rec["content_type"], t["cover_minutes"], t["understand_minutes"],', 'rec["difficulty"], rec["content_type"], t["cover_minutes"], t["cover_minutes"],'),
    ("write: run id lost", 'rec["taxonomy_version"], rec["cache_key"], run_id,', 'rec["taxonomy_version"], rec["cache_key"], None,'),
    ("report: textless posts not counted", "report.items_without_text += rec is None", "report.items_without_text += 0"),
    ("report: posts with links not counted", 'report.resource_link_posts += bool(split_links(row.get("urls") or ""))', "report.resource_link_posts += 0"),
    ("report: noise posts not recognised", 'noise = rec["clustering_subtopic_id"] == I.OTHER_ID', "noise = False"),
    ("report: as-run flags not counted", 'report.needs_review_as_run += bool(rec["needs_review"])', "report.needs_review_as_run += 0"),
    ("report: flags now not counted", "report.needs_review_now += new[\"needs_review\"]", "report.needs_review_now += 0"),
    ("report: items not counted", "                report.items += 1", "                report.items += 0"),
    ("load: database errors not turned into readable ones", "except sqlite3.Error as exc:", "except ZeroDivisionError as exc:"),
    ("load: bad taxonomy crashes", "except (I.InputError, RV.ReviewRuleError) as exc:", "except ZeroDivisionError as exc:"),
    # --- checks after the load
    ("verify: missing item links not noticed", '"SELECT COUNT(*) FROM items WHERE post_url IS NULL OR post_url = \'\'"', '"SELECT COUNT(*) FROM items WHERE 0"'),
    ("verify: missing library links not noticed", '"SELECT COUNT(*) FROM library WHERE post_url IS NULL OR post_url = \'\'"', '"SELECT COUNT(*) FROM library WHERE 0"'),
    ("verify: links counted but not stored not noticed", "AND (resource_link IS NULL OR resource_links = '[]')", "AND 0"),
    ("verify: classified without active row not noticed", "(SELECT 1 FROM classifications c WHERE c.item_id = i.id AND c.is_active = 1)", "(SELECT 1 FROM classifications c WHERE 0)"),
    ("verify: two active rows not noticed", "HAVING COUNT(*) > 1", "HAVING COUNT(*) > 5"),
    ("verify: dangling pointers not noticed", "    if bad:\n", "    if False:\n"),
    ("verify: lost resource links not noticed", "        if kept != with_links:", "        if False:"),
    ("verify: not run against the CSV", "report.problems = verify(con, posts)", "report.problems = verify(con)"),
    # --- command line
    ("cli: bad input exits 1", 'print(f"CANNOT LOAD:\\n{exc}")\n        return 2', 'print(f"CANNOT LOAD:\\n{exc}")\n        return 1'),
    ("cli: failed checks exit 0", '            print(f"  - {p}")\n        return 1', '            print(f"  - {p}")\n        return 0'),
    ("cli: failed checks not printed", "    if rep.problems:\n", "    if False:\n"),
    ("cli: repeated lines not mentioned", "    if rep.duplicate_lines:", "    if False:"),
    ("cli: default database path", 'default="data/library.sqlite"', 'default="library.sqlite"'),
    ("cli: default rule not the shipped file", 'default=str(RV.DEFAULT_PATH)', "default=None"),
    ("cli: flags not shown", 'print(f"  needs review: {rep.needs_review_now} of {rep.records} now, {rep.needs_review_as_run} as the run wrote it")', "pass"),
    ("cli: noise numbers not shown", 'print(f"  clustering-noise posts:', 'print(f"  noise posts:'),
    # --- the table rules and the view
    ("table: difficulty not limited", "CHECK (difficulty IN ('beginner','intermediate','advanced','expert'))", ""),
    ("table: cover time may be zero", "CHECK (cover_minutes > 0)", "CHECK (cover_minutes >= 0)"),
    ("table: understand may be shorter", "CHECK (understand_minutes >= cover_minutes)", ""),
    ("table: score not limited", "CHECK (alignment_score BETWEEN 0 AND 10)", ""),
    ("table: empty post link allowed", "CHECK (post_url <> '')", ""),
    ("table: two posts may share a link", "post_url            TEXT NOT NULL UNIQUE", "post_url            TEXT NOT NULL"),
    ("table: cache key may repeat", "cache_key             TEXT NOT NULL UNIQUE,", "cache_key             TEXT NOT NULL,"),
    ("table: several active rows allowed", "CREATE UNIQUE INDEX ux_active_cls", "CREATE INDEX ux_active_cls"),
    ("table: reviews may point at nothing", "item_id           TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,\n  classification_id", "item_id           TEXT NOT NULL,\n  classification_id"),
    ("table: items keep their text when deleted", "item_id       TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,", "item_id       TEXT PRIMARY KEY REFERENCES items(id),"),
    ("view: older rows shown too", "JOIN classifications c ON c.item_id = i.id AND c.is_active = 1", "JOIN classifications c ON c.item_id = i.id"),
    ("view: duplicates shown", "WHERE i.duplicate_of IS NULL;", ";"),
    ("view: post link missing", "       i.post_url, i.resource_link, i.resource_links, i.resource_link_count,", "       i.resource_link, i.resource_links, i.resource_link_count,"),
    ("view: first resource link missing", "i.post_url, i.resource_link, i.resource_links,", "i.post_url, i.resource_links,"),
    ("view: all resource links missing", "i.resource_link, i.resource_links, i.resource_link_count,", "i.resource_link, i.resource_link_count,"),
    ("view: topic names missing", "t.name AS topic_name, ", "NULL AS topic_name, "),
    ("view: subtopic names missing", "s.name AS subtopic_name,", "NULL AS subtopic_name,"),
    ("view: names from any taxonomy version", "LEFT JOIN topics t ON t.taxonomy_version = c.taxonomy_version AND t.id = c.topic_id", "LEFT JOIN topics t ON t.id = c.topic_id"),
]

if __name__ == "__main__":
    sys.exit(run_mutations(T, TESTS, MUTATIONS))
