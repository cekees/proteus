#!/usr/bin/env python3
"""Git filter that stores notebooks without outputs, run counts or kernel metadata (see .gitattributes).

    python3 tools/strip_notebook.py < in.ipynb > out.ipynb   # the clean filter
    python3 tools/strip_notebook.py in.ipynb                 # the diff textconv
    python3 tools/strip_notebook.py --install                # point this clone's git at it

setup.py runs --install on every build from a git checkout, so nobody has to set it up by hand. Running a
cell then changes nothing git sees; editing one does.
"""
import json
import subprocess
import sys

COMMAND = "python3 tools/strip_notebook.py"
RUN_METADATA = ("kernelspec", "language_info", "widgets")
CELL_RUN_METADATA = ("execution", "ExecuteTime", "collapsed", "scrolled")


def strip(text):
    try:
        nb = json.loads(text)
        cells = nb["cells"]
    except (ValueError, KeyError, TypeError):
        return text  # not a notebook git can strip (e.g. merge conflict markers): leave it alone
    for key in RUN_METADATA:
        nb.get("metadata", {}).pop(key, None)
    for cell in cells:
        for key in CELL_RUN_METADATA:
            cell.get("metadata", {}).pop(key, None)
        if cell.get("cell_type") == "code":
            cell["outputs"], cell["execution_count"] = [], None
    # the layout Jupyter itself writes, so the stored file and a stripped working copy compare equal
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def install():
    changed = False
    for key in ("filter.notebook-strip.clean", "diff.notebook-strip.textconv"):
        current = subprocess.run(["git", "config", "--get", key], capture_output=True, text=True).stdout.strip()
        if current != COMMAND:
            subprocess.run(["git", "config", key, COMMAND], check=True)
            changed = True
    if changed:
        print("git now stores notebooks without outputs (.gitattributes, tools/strip_notebook.py)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--install"]:
        install()
    else:
        text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
        sys.stdout.write(strip(text))
