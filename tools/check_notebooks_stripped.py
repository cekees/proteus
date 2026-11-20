#!/usr/bin/env python
"""Fail if a notebook under the given directories was committed with outputs or run metadata.

    python tools/check_notebooks_stripped.py notebooks

Notebooks are stored stripped (see .gitattributes): no outputs, no execution counts, no
kernelspec or language_info. A clone that was never built (so never ran
`python3 tools/strip_notebook.py --install`) commits them unstripped without any warning from
git; this check is what catches it in CI. Fix: run that command, then
`git add --renormalize notebooks` and commit again.
"""
import json
import pathlib
import sys

RUN_METADATA = ("kernelspec", "language_info", "widgets")


def problems(path):
    nb = json.loads(path.read_text())
    found = [f"notebook metadata '{k}'" for k in RUN_METADATA if k in nb.get("metadata", {})]
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            found.append(f"cell {i}: outputs")
        if cell.get("execution_count") is not None:
            found.append(f"cell {i}: execution count")
    return found


def main(dirs):
    notebooks = sorted(p for d in dirs for p in pathlib.Path(d).rglob("*.ipynb")
                       if ".ipynb_checkpoints" not in p.parts)
    failed = 0
    for path in notebooks:
        found = problems(path)
        if found:
            failed += 1
            print(f"{path}: not stripped ({'; '.join(found[:5])}{'; ...' if len(found) > 5 else ''})")
    print(f"{len(notebooks)} notebooks checked, {failed} not stripped")
    if failed:
        print("Fix: `python3 tools/strip_notebook.py --install`, then `git add --renormalize notebooks` and commit.")
    return 1 if failed or not notebooks else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["notebooks"]))
