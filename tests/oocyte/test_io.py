import tempfile
import unittest
from pathlib import Path

import numpy as np

from aegle.oocyte import DONOR13_V6, find_channel_index
from aegle.oocyte.io import (
    extract_padded_patch,
    load_candidate_mask,
    save_candidate_mask,
)
from aegle.oocyte.models import BoundingBox
from aegle.oocyte.segmentation import segment_oocyte_patch

from tests.oocyte.test_segmentation import circular_patch


class TestOocyteIO(unittest.TestCase):
    def test_finds_channel_index_from_explicit_channel_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "antibodies.tsv"
            path.write_text(
                "version\tchannel_id\tantibody_name\n"
                "2\tChannel:0:0\tDAPI\n"
                "2\tChannel:0:29\tUCHL1\n"
            )
            self.assertEqual(find_channel_index(path), 29)

    def test_extracts_and_unpads_edge_patch(self):
        image = np.arange(80, dtype=np.uint16).reshape(8, 10)
        extracted = extract_padded_patch(image, center_xy=(0, 0), radius=2)

        self.assertEqual(extracted.image.shape, (5, 5))
        self.assertEqual(extracted.bbox, BoundingBox(0, 0, 3, 3))
        self.assertEqual(extracted.padding_tblr, (2, 0, 2, 0))
        cropped = extracted.crop_to_image_bounds(np.ones((5, 5), dtype=np.bool_))
        self.assertEqual(cropped.shape, (3, 3))

    def test_extracts_patch_centered_just_outside_image(self):
        image = np.arange(80, dtype=np.uint16).reshape(8, 10)
        extracted = extract_padded_patch(image, center_xy=(-1, 3), radius=2)

        self.assertEqual(extracted.image.shape, (5, 5))
        self.assertEqual(extracted.bbox, BoundingBox(0, 1, 2, 6))
        self.assertEqual(extracted.padding_tblr, (0, 0, 3, 0))
        cropped = extracted.crop_to_image_bounds(np.ones((5, 5), dtype=np.bool_))
        self.assertEqual(cropped.shape, (5, 2))

    def test_candidate_mask_round_trip(self):
        result = segment_oocyte_patch(circular_patch(), DONOR13_V6)
        bbox = BoundingBox(100, 200, 461, 561)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_candidate.npz"
            save_candidate_mask(
                path,
                mask=result.mask,
                bbox=bbox,
                image_shape_yx=(1000, 1000),
                sample_id="13-23",
                candidate_id="13-23_det_0001",
                profile_name=DONOR13_V6.profile_name,
                profile_fingerprint=DONOR13_V6.fingerprint(),
                metrics=result.metrics,
            )
            loaded = load_candidate_mask(path)

        np.testing.assert_array_equal(loaded.mask, result.mask)
        self.assertEqual(loaded.bbox, bbox)
        self.assertEqual(loaded.image_shape_yx, (1000, 1000))
        self.assertEqual(loaded.metadata["sample_id"], "13-23")
        self.assertEqual(loaded.metadata["candidate_id"], "13-23_det_0001")
        self.assertEqual(
            loaded.metadata["profile_fingerprint"],
            DONOR13_V6.fingerprint(),
        )


if __name__ == "__main__":
    unittest.main()
