#!/usr/bin/env python
import sys
print("=== SCRIPT STARTED ===", flush=True)
sys.stdout.flush()
sys.stderr.flush()

print("About to import os", flush=True)
import os

print("About to load dotenv", flush=True)
from dotenv import load_dotenv
load_dotenv()

print("DATA DIR:", os.getenv("SOURCE_DATA_DIR"), flush=True)

print("About to import path_utils", flush=True)
from src.utils.path_utils import get_data_dir
print("Got data dir:", get_data_dir(), flush=True)

print("About to import torch", flush=True)
import torch
print("Torch imported", flush=True)

print("About to import pytorch_lightning", flush=True)
import pytorch_lightning as L
print("Lightning imported", flush=True)

print("=== ALL IMPORTS SUCCESSFUL ===", flush=True)