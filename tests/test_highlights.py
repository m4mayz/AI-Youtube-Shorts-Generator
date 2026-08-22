import unittest

from shorts_generator.highlights import chunk_transcript
from shorts_generator.local.transcriber import (
    _groq_segments,
    _is_retryable_groq_error,
    _transcript_cache_path,
)


class ChunkTranscriptTests(unittest.TestCase):
    def test_chunk_timestamps_are_relative_to_offset(self):
        transcript = {
            "duration": 2400,
            "segments": [
                {"start": 1145.0, "end": 1205.0, "text": "overlap"},
                {"start": 2285.0, "end": 2330.0, "text": "third chunk"},
            ],
        }

        chunks = chunk_transcript(transcript)

        self.assertEqual(chunks[1]["_offset"], 1140)
        self.assertEqual(chunks[1]["segments"][0]["start"], 5.0)
        self.assertEqual(chunks[1]["segments"][0]["end"], 65.0)
        self.assertEqual(chunks[2]["segments"][0]["start"], 5.0)
        self.assertEqual(chunks[2]["duration"], 50.0)

    def test_groq_segments_restore_chunk_offset(self):
        segments = _groq_segments(
            {"segments": [{"start": 2.5, "end": 4.0, "text": " hello "}]},
            1800,
        )

        self.assertEqual(
            segments,
            [{"start": 1802.5, "end": 1804.0, "text": "hello"}],
        )

    def test_groq_uses_a_separate_cache(self):
        self.assertEqual(_transcript_cache_path("video.mp4").name, "video.srt")
        self.assertEqual(_transcript_cache_path("video.mp4", "groq").name, "video.groq.srt")

    def test_groq_only_rotates_for_retryable_errors(self):
        rate_limit = RuntimeError("limited")
        rate_limit.status_code = 429
        bad_request = RuntimeError("bad request")
        bad_request.status_code = 400

        self.assertTrue(_is_retryable_groq_error(rate_limit))
        self.assertFalse(_is_retryable_groq_error(bad_request))


if __name__ == "__main__":
    unittest.main()
