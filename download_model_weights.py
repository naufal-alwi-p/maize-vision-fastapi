import gdown
import os
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--convnext-file-id", metavar="FILE_ID", required=True, help="File ID of ConvNeXt weights model stored in Google Drive")
parser.add_argument("--maxvit-file-id", metavar="FILE_ID", required=True, help="File ID of MaxViT weights model stored in Google Drive")

args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent
CONVNEXT_WEIGHTS = BASE_DIR / "model_weights" / "convnext_weights.pth"
MAXVIT_WEIGHTS = BASE_DIR / "model_weights" / "maxvit_weights.pth"

try:
    os.makedirs(BASE_DIR / "model_weights", exist_ok=True)
except Exception as e:
    print(e)

    sys.exit(1)

try:
    if not CONVNEXT_WEIGHTS.exists():
        gdown.download(id=args.convnext_file_id, output=str(CONVNEXT_WEIGHTS))
    
    if not MAXVIT_WEIGHTS.exists():
        gdown.download(id=args.maxvit_file_id, output=str(MAXVIT_WEIGHTS))
except Exception as e:
    print(e)

    sys.exit(1)
