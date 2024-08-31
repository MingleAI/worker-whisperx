INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'large-v2'
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'diarize': {
        'type': bool,
        'required': False,
        'default': False
    },
    'min_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    # Add other parameters as needed
}