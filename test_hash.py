from pathlib import Path

from utils.hash_utils import FileHasher

file = Path("data/all_documents/policy.md")

print(FileHasher.sha256(file))