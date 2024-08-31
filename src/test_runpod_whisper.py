import unittest
import json
import base64
from rp_handler import run_whisperx_job

class TestRunpodWhisper(unittest.TestCase):
    def test_run_whisper_job(self):
        # Read the sample audio file
        # with open("tests/sample_audio.wav", "rb") as audio_file:
            # audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Create a mock job input
        job_input = {
            "input": {
                "audio": "https://www.zamzar.com/download.php?uid=f37311e5b949941d121956e4eda1b-625c4c5ad874358&tcs=Z99&fileID=p1i6jiv4g74h416as7qn128a1884.wav",
                "diarize": True,
                "min_speakers": 1,
                "max_speakers": 5
            }
        }
        
        # Run the job
        result = run_whisperx_job(job_input)
        
        # Check the result
        self.assertIn('transcription', result)
        self.assertIsInstance(result['transcription'], str)
        self.assertGreater(len(result['transcription']), 0)

    def test_invalid_input(self):
        # Create an invalid job input
        job_input = {
            "input": {
                "invalid_key": "invalid_value"
            }
        }
        
        # Run the job
        result = run_whisper_job(job_input)
        
        # Check the result
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()