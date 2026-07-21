import gdown
import os
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--convnext-file-id", metavar="FILE_ID", required=True, help="File ID of ConvNeXt weights model stored in Google Drive")
# parser.add_argument("--maxvit-file-id", metavar="FILE_ID", required=True, help="File ID of MaxViT weights model stored in Google Drive")
parser.add_argument("--binary-file-id", metavar="FILE_ID", required=True, help="File ID of Binary model stored in Google Drive")

args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent
CONVNEXT_WEIGHTS = BASE_DIR / "model_weights" / "convnext_weights.pth"
# MAXVIT_WEIGHTS = BASE_DIR / "model_weights" / "maxvit_weights.pth"
BINARY_MODEL_PATH = BASE_DIR / "model_weights" / "binary.joblib"

try:
    os.makedirs(BASE_DIR / "model_weights", exist_ok=True)
except Exception as e:
    print(e)

    sys.exit(1)

try:
    if not CONVNEXT_WEIGHTS.exists():
        gdown.download(id=args.convnext_file_id, output=str(CONVNEXT_WEIGHTS))
    
    # if not MAXVIT_WEIGHTS.exists():
    #     gdown.download(id=args.maxvit_file_id, output=str(MAXVIT_WEIGHTS))

    if not BINARY_MODEL_PATH.exists():
        gdown.download(id=args.binary_file_id, output=str(BINARY_MODEL_PATH))
except Exception as e:
    print(e)

    sys.exit(1)
