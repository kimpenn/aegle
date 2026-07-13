import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from aegle.oocyte.recall_review import RecallReviewRuntime
from aegle.oocyte.recall_review_batch import (
    _batch_handler_for,
    generate_batch_recall_review_bundle,
)
from tests.oocyte.test_recall_review import RecallReviewFixture


class TestBatchRecallReview(unittest.TestCase):
    def test_generates_sample_consoles_and_identity_isolated_batch_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = RecallReviewFixture(root, "sample-a")
            second = RecallReviewFixture(root, "sample-b")
            for fixture in (first, second):
                (fixture.sample_dir / "oocytes.html").write_text(
                    f"<html><body>{fixture.sample_id} precision</body></html>"
                )
            (root / "oocyte_review_index.html").write_text("precision index")
            (root / "oocyte_detection_algorithm.html").write_text("algorithm")

            bundle = generate_batch_recall_review_bundle(
                root,
                sample_ids=["sample-a", "sample-b"],
            )

            self.assertEqual(bundle.sample_ids, ("sample-a", "sample-b"))
            self.assertEqual(bundle.total_candidate_count, 2)
            self.assertTrue(bundle.index_path.is_file())
            self.assertTrue(bundle.manifest_path.is_file())
            index = bundle.index_path.read_text()
            self.assertIn("sample-a/review_console.html", index)
            self.assertIn("sample-b/recall_review.html", index)
            self.assertIn("Review workflow", index)
            self.assertIn("Open console", index)
            self.assertIn("notes/oocytes_detection/reviews/", index)
            manifest = json.loads(bundle.manifest_path.read_text())
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(
                [row["sample_id"] for row in manifest["samples"]],
                ["sample-a", "sample-b"],
            )
            for fixture in (first, second):
                console = fixture.sample_dir / "review_console.html"
                self.assertTrue(console.is_file())
                self.assertIn(fixture.sample_id, console.read_text())

            with ExitStack() as stack:
                runtimes = {
                    sample_id: stack.enter_context(
                        RecallReviewRuntime(root / sample_id)
                    )
                    for sample_id in bundle.sample_ids
                }
                server = ThreadingHTTPServer(
                    ("127.0.0.1", 0),
                    _batch_handler_for(runtimes, bundle),
                )
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                try:
                    batch_health = json.loads(
                        urlopen(base + "/health", timeout=3).read()
                    )
                    self.assertEqual(
                        batch_health["sample_ids"], ["sample-a", "sample-b"]
                    )
                    root_page = urlopen(base + "/", timeout=3).read().decode()
                    self.assertIn("Oocyte review consoles", root_page)
                    with urlopen(base + "/sample-a", timeout=3) as response:
                        self.assertTrue(response.geturl().endswith("/sample-a/"))
                        self.assertIn(
                            "sample-a oocyte review console",
                            response.read().decode(),
                        )
                    sample_health = json.loads(
                        urlopen(base + "/sample-b/health", timeout=3).read()
                    )
                    self.assertEqual(sample_health["sample_id"], "sample-b")
                    metadata = json.loads(
                        urlopen(base + "/sample-a/api/metadata", timeout=3).read()
                    )
                    self.assertEqual(metadata["sample_id"], "sample-a")
                    patch = urlopen(
                        base
                        + "/sample-b/api/patch.webp?x=380&y=380&radius=128",
                        timeout=3,
                    ).read()
                    self.assertGreater(len(patch), 1000)
                    precision = urlopen(
                        base + "/sample-a/oocytes.html", timeout=3
                    ).read()
                    self.assertIn(b"sample-a precision", precision)
                    with self.assertRaises(HTTPError) as context:
                        urlopen(base + "/not-a-sample/api/metadata", timeout=3)
                    self.assertEqual(context.exception.code, 404)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)

    def test_rejects_duplicate_and_unsafe_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RecallReviewFixture(root, "sample-a")
            with self.assertRaisesRegex(ValueError, "must be unique"):
                generate_batch_recall_review_bundle(
                    root,
                    sample_ids=["sample-a", "sample-a"],
                )
            with self.assertRaisesRegex(ValueError, "invalid batch sample ID"):
                generate_batch_recall_review_bundle(
                    root,
                    sample_ids=["../sample-a"],
                )


if __name__ == "__main__":
    unittest.main()
