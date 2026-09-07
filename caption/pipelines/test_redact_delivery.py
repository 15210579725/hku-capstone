import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from redact_delivery import Audit, main, sanitize_obj


class PersonNameRedactionTests(unittest.TestCase):
    def sanitize(self, obj):
        audit = Audit()
        clean = sanitize_obj(obj, "sample/captions.jsonl", 1, audit)
        return clean, audit

    def test_speaker_prefix_name_is_redacted_everywhere_in_record(self):
        source = {
            "ok": True,
            "model": "gemini-3.7-flash",
            "parsed": {
                "text_visible": ["Gemini: ask a question"],
                "speech": "Jake: pass it around",
                "action": "Jake让我们戳一下手机",
                "details": "Transcript: Jake speaks.",
            },
        }

        clean, audit = self.sanitize(source)
        rendered = json.dumps(clean, ensure_ascii=False)

        self.assertNotIn("Jake", rendered)
        self.assertIn("Gemini: ask a question", rendered)
        self.assertEqual(clean["model"], "gemini-3.7-flash")
        self.assertIn("Transcript:", rendered)
        self.assertGreaterEqual(audit.counts["person_name_in_text"], 3)

    def test_sensitive_name_field_seeds_redaction_in_other_strings(self):
        source = {
            "ok": True,
            "parsed": {
                "speaker": "Alice",
                "details": "Alice opens the laptop.",
            },
        }

        clean, audit = self.sanitize(source)
        rendered = json.dumps(clean, ensure_ascii=False)

        self.assertNotIn("Alice", rendered)
        self.assertEqual(clean["parsed"]["speaker"], "XXX")
        self.assertIn("XXX opens the laptop.", rendered)
        self.assertGreaterEqual(audit.counts["person_name_in_text"], 1)

    def test_single_letter_speaker_role_does_not_redact_normal_text(self):
        source = {
            "ok": True,
            "parsed": {
                "speaker": "A",
                "details": "A person opens a door.",
            },
        }

        clean, _ = self.sanitize(source)
        self.assertEqual(clean["parsed"]["speaker"], "XXX")
        self.assertEqual(clean["parsed"]["details"], "A person opens a door.")

    def test_me_speaker_role_does_not_redact_pronoun(self):
        source = {
            "ok": True,
            "parsed": {
                "speaker": "Me",
                "details": "Tell me what happened.",
            },
        }

        clean, _ = self.sanitize(source)
        self.assertEqual(clean["parsed"]["speaker"], "XXX")
        self.assertEqual(clean["parsed"]["details"], "Tell me what happened.")

    def test_lite_role_does_not_corrupt_model_identifier(self):
        source = {
            "ok": True,
            "model": "gemini-3.5-flash-lite",
            "parsed": {
                "speaker": "Lite",
                "details": "The lite model remains selected.",
            },
        }

        clean, _ = self.sanitize(source)

        self.assertEqual(clean["parsed"]["speaker"], "XXX")
        self.assertEqual(clean["model"], "gemini-3.5-flash-lite")
        self.assertEqual(clean["parsed"]["details"], "The lite model remains selected.")

    def test_email_with_trailing_sentence_punctuation_is_redacted(self):
        clean, audit = self.sanitize({
            "ok": True,
            "parsed": {"text_visible": ["Contact test.person@example.com."]},
        })

        rendered = json.dumps(clean, ensure_ascii=False)
        self.assertNotIn("test.person@example.com", rendered)
        self.assertIn("Contact XXX.", rendered)
        self.assertEqual(audit.counts["email"], 1)

    def test_huggingface_token_is_redacted(self):
        token = "hf_abcdefghijklmnopqrstuvwxyz"
        clean, audit = self.sanitize({
            "ok": True,
            "parsed": {"text_visible": [f"token={token}"]},
        })

        rendered = json.dumps(clean, ensure_ascii=False)
        self.assertNotIn(token, rendered)
        self.assertIn("token=XXX", rendered)
        self.assertEqual(audit.counts["api_token"], 1)

    def test_raw_model_duplicate_is_removed_but_parsed_result_is_kept(self):
        clean, audit = self.sanitize({
            "ok": True,
            "content_raw": '{"scene_summary":"private duplicate"}',
            "parsed": {"scene_summary": "safe structured result"},
        })

        self.assertEqual(clean["content_raw"], "XXX")
        self.assertEqual(clean["parsed"]["scene_summary"], "safe structured result")
        self.assertEqual(audit.counts["raw_duplicate_removed"], 1)


class DeliveryIndexTests(unittest.TestCase):
    def test_index_is_recomputed_from_kept_redacted_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            dst = root / "dst"
            report = root / "audit.json"
            rec = src / "recording-a"
            rec.mkdir(parents=True)
            rows = [
                {
                    "ok": True,
                    "clip_id": 1,
                    "model": "model-a",
                    "parsed": {"segments": [{"action": "walk"}]},
                },
                {"ok": False, "clip_id": 2, "model": "model-b"},
            ]
            (rec / "captions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (rec / "captions.txt").write_text(
                "===== clip 1 =====\nwalk\n===== clip 2 =====\nfailed\n",
                encoding="utf-8",
            )
            (src / "index.json").write_text(
                json.dumps(
                    {
                        "recording-a": {
                            "usable_clips": 2,
                            "target_clips": 1,
                            "minutes": 1,
                            "coverage": 1.0,
                            "in_first_100h": True,
                            "models": {"model-a": 1, "model-b": 1},
                            "segments": 1,
                        },
                        "_summary": {
                            "usable_clips_total": 2,
                            "first_100h_usable": 2,
                            "first_100h_target": 2,
                            "first_100h_coverage": 1.0,
                            "recordings_with_output": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "redact_delivery.py",
                "--src",
                str(src),
                "--dst",
                str(dst),
                "--report",
                str(report),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()

            index = json.loads((dst / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["recording-a"]["usable_clips"], 1)
            self.assertEqual(index["recording-a"]["models"], {"model-a": 1})
            self.assertEqual(index["recording-a"]["segments"], 1)
            self.assertEqual(index["recording-a"]["source_planned_target_clips"], 1)
            self.assertEqual(index["recording-a"]["selected_candidate_clips"], 2)
            self.assertEqual(index["recording-a"]["target_clips"], 2)
            self.assertEqual(index["recording-a"]["coverage"], 0.5)
            self.assertEqual(index["_summary"]["usable_clips_total"], 1)
            self.assertEqual(index["_summary"]["first_100h_usable"], 1)
            self.assertEqual(index["_summary"]["first_100h_target"], 2)
            self.assertEqual(index["_summary"]["first_100h_coverage"], 0.5)
            self.assertEqual(index["_summary"]["selected_candidate_clips_total"], 2)
            self.assertEqual(index["_summary"]["source_planned_target_clips"], 1)
            self.assertEqual(index["_summary"]["recordings_with_output"], 1)

    def test_name_signal_propagates_across_caption_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            dst = root / "dst"
            report = root / "audit.json"
            rec = src / "recording-a"
            rec.mkdir(parents=True)
            rows = [
                {
                    "ok": True,
                    "clip_id": 1,
                    "model": "gemini-3.7-flash",
                    "parsed": {"speaker": "Alice", "segments": [{"speech": "Jake: hello"}]},
                },
                {
                    "ok": True,
                    "clip_id": 2,
                    "model": "gemini-3.7-flash",
                    "parsed": {"segments": [{"text_visible": ["I heard Jake说话"]}]},
                },
                {
                    "ok": True,
                    "clip_id": 3,
                    "model": "gemini-3.7-flash",
                    "parsed": {"segments": [{"speech": "Current: status"}]},
                },
                {
                    "ok": True,
                    "clip_id": 4,
                    "model": "gemini-3.7-flash",
                    "parsed": {"segments": [{"details": "Current remains visible."}]},
                },
            ]
            (rec / "captions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (rec / "captions.txt").write_text(
                "===== clip 1 =====\nAlice and Jake: hello\n\n===== clip 2 =====\nI heard Jake说话\n\n===== clip 3 =====\nCurrent: status\n\n===== clip 4 =====\nCurrent remains visible.\n\n",
                encoding="utf-8",
            )
            (src / "index.json").write_text(
                json.dumps({
                    "recording-a": {
                        "usable_clips": 4,
                        "target_clips": 4,
                        "in_first_100h": True,
                    },
                    "_summary": {},
                }),
                encoding="utf-8",
            )

            argv = [
                "redact_delivery.py", "--src", str(src), "--dst", str(dst),
                "--report", str(report),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()

            rendered = "".join(
                (dst / "recording-a" / "captions.jsonl").read_text(encoding="utf-8").splitlines()
            )
            txt = (dst / "recording-a" / "captions.txt").read_text(encoding="utf-8")
            self.assertNotIn("Jake", rendered)
            self.assertNotIn("Jake", txt)
            self.assertNotIn("Alice", rendered)
            self.assertNotIn("Alice", txt)
            self.assertIn("Current remains visible.", rendered)
            self.assertIn("Current remains visible.", txt)
            self.assertFalse(txt.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
