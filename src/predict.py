import logging
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import whisperx
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Predictor:
    def __init__(self):
        self.models = {}
        self.align_models = {}
        self.diarization_pipeline = None

    def setup(self):
        logging.info("Starting setup")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        model_names = ["tiny"]
        
        for model_name in model_names:
            logging.info(f"Loading model: {model_name}")
            self.models[model_name] = whisperx.load_model(model_name, device, compute_type=compute_type,
                asr_options={
                    "max_new_tokens": 128,
                    "clip_timestamps": True,
                    "hallucination_silence_threshold": 2.0
                })
        
        logging.info("Loading diarization pipeline")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        if device == "cuda":
            self.diarization_pipeline.to(torch.device("cuda"))
        logging.info("Setup completed")

    def predict(
        self,
        audio,
        model="tiny",
        language=None,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
        num_speakers=None,
        batch_size=16,
        **kwargs
    ):
        logging.info(f"Starting prediction with model: {model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_data = self.models[model]
        
        logging.info("Loading audio")
        audio_data = whisperx.load_audio(audio)

        logging.info("Transcribing")
        transcribe_params = {
            "language": language,
            "batch_size": batch_size
        }
        # transcribe_params.update(kwargs)  # Добавляем все дополнительные параметры
        result = model_data.transcribe(audio_data, batch_size=batch_size, **kwargs)

        # result = model_data.transcribe(audio_data, **transcribe_params)
        logging.debug(f"Transcription result: {json.dumps(result, indent=2)}")
        
        if language is None:
            language = result["language"]
        logging.info(f"Using language: {language}")
        
        if language not in self.align_models:
            logging.info(f"Loading alignment model for language: {language}")
            self.align_models[language], metadata = whisperx.load_align_model(language_code=language, device=device)
        
        align_model = self.align_models[language]
        logging.info("Aligning")
        try:
            result = whisperx.align(result["segments"], align_model, metadata, audio_data, device, return_char_alignments=False)
            logging.debug(f"Alignment result: {json.dumps(result, indent=2)}")
        except Exception as e:
            logging.error(f"Error during alignment: {str(e)}")
            raise
        
        if diarize:
            logging.info("Starting diarization")
            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs['num_speakers'] = num_speakers
            elif min_speakers is not None or max_speakers is not None:
                if min_speakers is not None:
                    diarize_kwargs['min_speakers'] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs['max_speakers'] = max_speakers
            
            try:
                diarization = self.diarization_pipeline(audio, **diarize_kwargs)
                logging.info(f"Diarization result type: {type(diarization)}")
                logging.info(f"Diarization result: {diarization}")
                
                # Преобразование результатов диаризации в формат, ожидаемый whisperx
                diarize_dict = self.convert_diarization_to_dict(diarization)
                logging.info(f"Converted diarization dataframe: {diarize_dict}")
                
                logging.info("Assigning word speakers")
                result = whisperx.assign_word_speakers(diarize_dict, result)
                logging.info(f"Result after speaker assignment: {json.dumps(result, indent=2)}")

                diarization = self.diarization_pipeline(audio, **diarize_kwargs)
                logging.info(f"Diarization result: {diarization}")
                
                logging.info("Assigning word speakers")
                result = whisperx.assign_word_speakers(diarization, result)
                logging.info(f"Result after speaker assignment: {json.dumps(result, indent=2)}")
            except Exception as e:
                logging.error(f"Error during diarization or speaker assignment: {str(e)}")
                raise
        
        logging.info("Formatting result")
        return self.format_result(result)

    def convert_diarization_to_dict(self, diarization):
        segments = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': speaker,
                'word': ''  # Добавляем пустое слово, так как это требуется для assign_word_speakers
            })
        
        # Сортируем сегменты по времени начала
        segments.sort(key=lambda x: x['start'])
        
        diarize_dict = {
            'segments': segments,
            'word_segments': segments  # whisperx может ожидать это поле
        }
        
        logging.debug(f"Converted diarization dict: {json.dumps(diarize_dict, indent=2)}")
        return diarize_dict

    def format_result(self, result):
        try:
            formatted_result = {
                "segments": [],
                "language": result.get("language")
            }

            for segment in result["segments"]:
                formatted_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": segment.get("speaker"),
                    "words": [
                        {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "speaker": word.get("speaker")
                        }
                        for word in segment.get("words", [])
                    ]
                }
                formatted_result["segments"].append(formatted_segment)

            logging.debug(f"Formatted result: {json.dumps(formatted_result, indent=2)}")
            return formatted_result
        except Exception as e:
            logging.error(f"Error during result formatting: {str(e)}")
            raise

logging.info("Initializing MODEL")
MODEL = Predictor()
logging.info("Setting up MODEL")
MODEL.setup()
logging.info("MODEL setup complete")