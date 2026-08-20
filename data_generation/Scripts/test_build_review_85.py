#!/usr/bin/env python3
"""Regression tests for the restart-safe 85-video review renderer."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_review_85 as review


class Review85RendererTests(unittest.TestCase):
    def test_existing_output_directory_is_reused_without_clearing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "videos"
            output.mkdir()
            sentinel = output / "partial.mp4"
            sentinel.write_bytes(b"partial")

            self.assertEqual(review.prepare_output_directory(root), output)
            self.assertEqual(sentinel.read_bytes(), b"partial")

    def test_render_passes_the_bundled_ffmpeg_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = review.prepare_output_directory(root)
            ffmpeg = root / "ffmpeg.exe"

            with (
                patch.object(review.imageio_ffmpeg, "get_ffmpeg_exe", return_value=str(ffmpeg)),
                patch.object(
                    review.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 1),
                ) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "rendering failed"):
                    review.render(root / "dataset", ["episode-1"], output, "test")

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--ffmpeg") + 1], str(ffmpeg))


if __name__ == "__main__":
    unittest.main()
