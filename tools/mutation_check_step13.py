"""Mutation check for the Step 13 changes. Break one line at a time; the tests must notice.
Run:  python tools/mutation_check_step13.py        (exit 0 = all killed in all four files)

What Step 13 added, and where it is checked:
  * pipeline/review_rule.py      the review rule file and the decision      tests/test_review_rule.py
  * pipeline/classifier.py       limits, timeout, review fields, no re-ask  tests/test_classifier.py
  * pipeline/step9_classify.py   --num-predict, --timeout, --review-rule    tests/test_step9_classify.py
  * pipeline/ollama_json.py      the timeout error type                     tests/test_ollama_json.py

Not listed on purpose (no test could ever tell the difference):
  * the `# the None checks only help the type checker` half of the conditions in review_rule.py
    (`data is None or sha is None` can never be true when `problems` is empty).
  * a separate nan / infinity check for confidence_below: the range test `0 <= x <= 1` already
    rejects both (a first version had the extra check and its mutation survived, so it was removed).
"""
import sys

from mutation_runner import run_mutations

REVIEW_RULE = [
    ("rule file: not a mapping accepted", 'if not isinstance(data, dict):\n        return ["the file must contain', 'if False:\n        return ["the file must contain'),
    ("rule file: missing keys not reported", "    if missing:\n", "    if False:\n"),
    ("rule file: unknown keys (typos) not reported", "    if unknown:", "    if False:"),
    ("rule file: version not checked", 'if "version" in data and (not isinstance(version, str) or not version.strip()):', "if False:"),
    ("rule file: blank version accepted", " or not version.strip()", ""),
    ("rule file: true accepted as a number", " or isinstance(cb, bool)", ""),
    ("rule file: negative number accepted", "or not 0 <= cb <= 1)", "or not -1 <= cb <= 1)"),
    ("rule file: number above 1 accepted", "or not 0 <= cb <= 1)", "or not 0 <= cb <= 5)"),
    ("rule file: switches not checked", "if key in data and not isinstance(data[key], bool):", "if False:"),
    ("rule file: unreadable file crashes", 'return None, None, [f"cannot read {path}: {exc.strerror or exc}"]', "return None, None, []"),
    ("rule file: bad encoding crashes", "except (UnicodeDecodeError, yaml.YAMLError) as exc:", "except yaml.YAMLError as exc:"),
    ("rule file: fingerprint is a constant", "return data, rubrics_sha(raw), validate_review_rule(data)", 'return data, "0" * 64, validate_review_rule(data)'),
    ("load: problems do not stop it", "if problems or data is None or sha is None:     # the None checks only help the type checker\n        raise ReviewRuleError", "if False:\n        raise ReviewRuleError"),
    ("load: number kept as it was written", 'confidence_below=float(data["confidence_below"])', 'confidence_below=data["confidence_below"]'),
    ("main: problems not printed", 'if problems or data is None or sha is None:     # the None checks only help the type checker\n        print(f"PROBLEMS in', 'if False:\n        print(f"PROBLEMS in'),
    ("main: problems exit 0", '            print(f"  - {p}")\n        return 1', '            print(f"  - {p}")\n        return 0'),
    ("decision: boundary included", "if confidence < rule.confidence_below:", "if confidence <= rule.confidence_below:"),
    ("decision: low confidence never flagged", "if confidence < rule.confidence_below:", "if False:"),
    ("decision: other.general switch ignored", "if rule.other_general and subtopic_id == OTHER_ID:", "if subtopic_id == OTHER_ID:"),
    ("decision: other.general never flagged", "if rule.other_general and subtopic_id == OTHER_ID:", "if False:"),
    ("decision: disagreement switch ignored", "if (rule.topic_disagrees and clustering_subtopic_id not in", "if (clustering_subtopic_id not in"),
    ("decision: disagreement never flagged", "if (rule.topic_disagrees and clustering_subtopic_id not in", "if (False and clustering_subtopic_id not in"),
    ("decision: a post the clustering could not place counts", "clustering_subtopic_id not in (None, OTHER_ID)", "clustering_subtopic_id is not None"),
    ("decision: a post with no clustering opinion counts", "clustering_subtopic_id not in (None, OTHER_ID)", "clustering_subtopic_id != OTHER_ID"),
    ("decision: subtopic compared instead of topic", "taxonomy.topic_of(subtopic_id) != taxonomy.topic_of(clustering_subtopic_id)", "subtopic_id != clustering_subtopic_id"),
    ("decision: reasons in another order", "REASONS = (REASON_LOW_CONFIDENCE, REASON_OTHER, REASON_DISAGREES)", "REASONS = (REASON_OTHER, REASON_LOW_CONFIDENCE, REASON_DISAGREES)"),
    ("decision: reason renamed", 'REASON_OTHER = "other_general"', 'REASON_OTHER = "other"'),
    ("default path: wrong file", '"review_rule.v1.yaml"', '"review_rule.v2.yaml"'),
]

CLASSIFIER = [
    ("limits: call A default changed", "NUM_PREDICT_A = 400 ", "NUM_PREDICT_A = 4000 "),
    ("limits: call B default changed", "NUM_PREDICT_B = 1500 ", "NUM_PREDICT_B = 15000 "),
    ("limits: timeout default changed", "TIMEOUT_SECONDS = 300 ", "TIMEOUT_SECONDS = 3000 "),
    ("context: limits not checked", "if not _is_int(value) or value < 1:", "if False:"),
    ("context: zero accepted as a limit", "or value < 1:", "or value < 0:"),
    ("context: true accepted as a number", "if not _is_int(value) or value < 1:", "if not isinstance(value, int) or value < 1:"),
    ("context: shipped review rule not loaded", "if self.review is None:\n            self.review = RV.load_review_rule()", "if False:\n            self.review = RV.load_review_rule()"),
    ("ask: timeout is retried", 'except LLMTimeout as exc:\n            raise ClassifyFailed(f"timeout: {exc}")', "except LLMTimeout as exc:\n            answer, hard, soft = None, [str(exc)], []"),
    ("ask: timeout reason lost", 'raise ClassifyFailed(f"timeout: {exc}")', 'raise ClassifyFailed(f"{exc}")'),
    ("ask: timeout not sent to the model", "timeout=ctx.timeout, options=ctx.options)", "options=ctx.options)"),
    ("classify: call A limit ignored", "ctx.schema_a,\n                                    ctx.num_predict_a,", "ctx.schema_a,\n                                    400,"),
    ("classify: call B limit ignored", "ctx.num_predict_b, lambda", "1500, lambda"),
    ("check_b: a stated size that repair can fix asks again", '    soft.extend(_summary_problems(answer["summary"], rub))\n', '    soft.extend(_summary_problems(answer["summary"], rub))\n    if answer["stated_size"].strip() and answer["basis"] != "stated_in_post":\n        soft.append("x")\n'),
    ("record: review flag never set", '"needs_review": bool(reasons),', '"needs_review": False,'),
    ("record: review reasons not stored", '"review_reasons": reasons,', '"review_reasons": [],'),
    ("record: review rule version not stored", '"review_rule_version": ctx.review.version,', '"review_rule_version": "",'),
    ("record: review rule fingerprint not stored", '"review_rule_sha": ctx.review.sha,', '"review_rule_sha": "",'),
    ("classify: review does not see the clustering", "clustering_subtopic_id=clustering, taxonomy=ctx.taxonomy)", "clustering_subtopic_id=None, taxonomy=ctx.taxonomy)"),
    ("classify: review sees the clustering's subtopic as the model's", "confidence=conf, subtopic_id=subtopic,", "confidence=conf, subtopic_id=clustering or subtopic,"),
    ("classify: review sees a perfect confidence", "confidence=conf, subtopic_id=subtopic,", "confidence=1.0, subtopic_id=subtopic,"),
]

STEP9 = [
    ("cli: --num-predict ignored for call A", "num_predict_a=args.num_predict or C.NUM_PREDICT_A,", "num_predict_a=C.NUM_PREDICT_A,"),
    ("cli: --num-predict ignored for call B", "num_predict_b=args.num_predict or C.NUM_PREDICT_B,", "num_predict_b=C.NUM_PREDICT_B,"),
    ("cli: --timeout ignored", "timeout=args.timeout, review=review)", "review=review)"),
    ("cli: --timeout default changed", "default=C.TIMEOUT_SECONDS,", "default=3,"),
    ("cli: review rule not passed on", "timeout=args.timeout, review=review)", "timeout=args.timeout)"),
    ("cli: --review-rule file not read", "review = RV.load_review_rule(args.review_rule)", "review = RV.load_review_rule()"),
    ("cli: a bad review rule is not caught", "except (R.RubricsError, RV.ReviewRuleError, I.InputError) as exc:", "except (R.RubricsError, I.InputError) as exc:"),
    ("cli: zero accepted as a limit", "    if value < 1:\n", "    if value < 0:\n"),
    ("cli: text for a limit not explained", "    except ValueError:\n        raise argparse.ArgumentTypeError", "    except KeyError:\n        raise argparse.ArgumentTypeError"),
    ("summary: flagged posts not counted", 'flagged = sum(r["needs_review"] for r in stored)', "flagged = 0"),
    ("summary: reasons not listed", "if why else ''}", "if False else ''}"),
    ("summary: reasons counted per run instead of per reason", 'sum(name in r["review_reasons"] for r in stored)', "sum(1 for r in stored)"),
]

OLLAMA_JSON = [
    ("timeout: plain error instead of the timeout type", 'raise LLMTimeout(f"No answer within {timeout} seconds")', 'raise LLMError(f"No answer within {timeout} seconds")'),
    ("timeout: type is not an LLMError", "class LLMTimeout(LLMError):", "class LLMTimeout(Exception):"),
    ("timeout: limit not named", "No answer within {timeout} seconds", "No answer in time"),
    ("timeout: not sent to the request", 'requests.post(f"{url}/api/chat", json=payload, timeout=timeout)', 'requests.post(f"{url}/api/chat", json=payload)'),
]

GROUPS = [
    ("pipeline/review_rule.py", "tests/test_review_rule.py", REVIEW_RULE),
    ("pipeline/classifier.py", "tests/test_classifier.py", CLASSIFIER),
    ("pipeline/step9_classify.py", "tests/test_step9_classify.py", STEP9),
    ("pipeline/ollama_json.py", "tests/test_ollama_json.py", OLLAMA_JSON),
]

if __name__ == "__main__":
    worst = 0
    for target, test_file, mutations in GROUPS:
        print(f"\n=== {target}  ({len(mutations)} mutations, tests: {test_file})")
        worst = max(worst, run_mutations(target, test_file, mutations))
    print("\nALL KILLED" if worst == 0 else "\nSOME SURVIVED OR WERE UNUSABLE")
    sys.exit(worst)
