#!/usr/bin/env python3
"""Focused regression tests for Stage65B-R2 report parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage65b_r2_common import MAX_CSV_FIELD_SIZE, read_tsv


class CsvFieldLimitTest(unittest.TestCase):
    def parse_payload(self, size: int) -> list[dict[str, str]]:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "field.tsv"
            path.write_text("name\tpayload\nrow\t" + ("x" * size) + "\n", encoding="utf-8")
            return read_tsv(path)

    def test_129_kib_field_passes(self) -> None:
        self.assertEqual(len(self.parse_payload(129 * 1024)), 1)

    def test_exact_16_mib_field_passes(self) -> None:
        self.assertEqual(len(self.parse_payload(MAX_CSV_FIELD_SIZE)), 1)

    def test_over_16_mib_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds the 16777216-character limit"):
            self.parse_payload(MAX_CSV_FIELD_SIZE + 1)


class BootstrapEnvelopeTest(unittest.TestCase):
    def test_vectorized_reverse_envelope_is_byte_identical(self) -> None:
        rng = np.random.default_rng(65002)
        for length in (1, 2, 101, 10000):
            original = rng.random(length, dtype=np.float64)
            expected = original.copy()
            for index in range(length - 1, 0, -1):
                if expected[index] > expected[index - 1]:
                    expected[index - 1] = expected[index]
            observed = np.maximum.accumulate(original[::-1])[::-1]
            self.assertTrue(np.array_equal(expected, observed))


if __name__ == "__main__":
    unittest.main()
