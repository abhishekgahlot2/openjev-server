"""Mirror this repository to the Hugging Face Space openjev/openjev-server. The Space wants YAML front matter at the top of README.md
and GitHub renders that block as a table, so the front matter lives here and is prepended at upload. HF_TOKEN from the environment."""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

SPACE = "openjev/openjev-server"
FRONT_MATTER = """---
title: openjev-server
emoji: ⚖️
colorFrom: green
colorTo: gray
sdk: static
pinned: true
license: apache-2.0
short_description: A decision API over any open model, one pass per question
---

"""


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    files = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    ops = []
    for f in files:
        if f == "README.md":
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=(FRONT_MATTER + (root / f).read_text()).encode()))
        else:
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(root / f)))
    last = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%s"], capture_output=True, text=True).stdout.strip()
    message = sys.argv[1] if len(sys.argv) > 1 else last
    info = HfApi(token=os.environ["HF_TOKEN"]).create_commit(repo_id=SPACE, repo_type="space", operations=ops, commit_message=message)
    print(f"published {len(files)} files to {SPACE}: {info.oid[:8]}")


if __name__ == "__main__":
    main()
