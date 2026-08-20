import unittest

from shorts_generator.highlights import chunk_transcript


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


if __name__ == "__main__":
    unittest.main()
