import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import machine_extractor


class MachineExtractorTests(unittest.TestCase):
    def test_extract_all_machines_handles_empty_input(self):
        self.assertEqual(machine_extractor.extract_all_machines(""), [])

    def test_extract_all_machines_handles_extraction_errors(self):
        def fail_chunk(_chunk):
            raise RuntimeError("boom")

        original = machine_extractor.extract_machines_from_chunk
        machine_extractor.extract_machines_from_chunk = fail_chunk
        try:
            self.assertEqual(machine_extractor.extract_all_machines("some methods text"), [])
        finally:
            machine_extractor.extract_machines_from_chunk = original

    def test_extract_machines_from_chunk_uses_local_backend_response(self):
        original = machine_extractor._query_model
        machine_extractor._query_model = lambda *args, **kwargs: machine_extractor.MachineExtraction(machines=["Illumina"])
        try:
            result = machine_extractor.extract_machines_from_chunk("some methods text")
            self.assertEqual(result.machines, ["Illumina"])
        finally:
            machine_extractor._query_model = original


if __name__ == "__main__":
    unittest.main()
