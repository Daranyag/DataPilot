import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SANDBOX_TEMP_DIR = BASE_DIR / "sandbox_temp"

# Ensure the temp directory exists
os.makedirs(SANDBOX_TEMP_DIR, exist_ok=True)

# Sandbox settings
TIMEOUT_SECONDS = 15
MAX_RETRIES = 3

# Blocked imports and builtins for security
BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "pathlib", "pty", "socket", 
    "requests", "urllib", "http", "ftplib"
}

BLOCKED_BUILTINS = {
    "eval", "exec", "open", "__import__", "globals", "locals", 
    "compile", "input"
}
