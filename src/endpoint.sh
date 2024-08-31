#!/bin/bash
if [ "$1" = "test" ]; then
  pytest test_runpod_whisper.py
else
  python rp_handler.py
fi
