"""The chain of YMF files a spec is built from, and the state of each.

``pro weak`` and ``pro dis`` each write a file that ``extends`` its input,
so a spec is a chain::

    poisson.ymf             written by a person
      └ poisson.weak.ymf    pro weak poisson.ymf
          └ poisson.disc.ymf    pro dis poisson.weak.ymf
    poisson.disc.archive.ymf    pro run poisson.disc.ymf

This module follows ``extends`` up from a file, finds the files beside it
that extend it, and says for each generated one whether it still holds
(``ymf.generated.check``): **stale** when what it read upstream changed,
**edited** when someone changed what it wrote. ``pro status`` reports the
chain; ``pro run`` refuses a stale one.
"""
import os

__all__ = ["file_info", "chain", "descendants", "stale", "regenerate_command"]

#: the commands that write generated files, as their records name them (a
#: record written before pro disc was renamed pro dis matches "pro dis" too)
_STEPS = ("pro weak", "pro dis")


def _read(path):
    """A file's own data (partial schema), without composing it."""
    from pathlib import Path
    from ymf.compose import _read as read
    data, _ = read(Path(os.path.abspath(path)))
    return data


def regenerate_command(path, record):
    """The command that writes ``path`` again, from its ``generated`` record."""
    data = _read(path)
    step = next((s for s in _STEPS if str(record.get("by", "")).startswith(s)), None)
    source = data.get("extends")
    if step is None or source is None:
        return None
    here = os.path.dirname(os.path.abspath(path))
    source = os.path.relpath(os.path.join(here, source))
    default = {"pro weak": ".weak.ymf", "pro dis": ".disc.ymf"}[step]
    from ymf.closure import archive_stem
    stem = archive_stem(source)
    if stem.endswith(".weak"):
        stem = stem[:-len(".weak")]
    out = "" if os.path.basename(path) == stem + default else " -o %s" % (
        os.path.relpath(path),)
    return "%s %s%s" % (step, source, out)


def file_info(path):
    """``{file, extends, declared, generated}`` for one file.

    ``generated`` is None for a file no tool wrote, else ``{by, stale,
    edited, current, reasons, regenerate}``; a malformed record gives
    ``{error}``.
    """
    from ymf import generated
    path = os.path.abspath(path)
    data = _read(path)
    info = {"file": path, "extends": data.get("extends"),
            "declared": data.get("ymf_state"), "generated": None}
    try:
        status = generated.check(path)
    except ValueError as exc:
        info["generated"] = {"error": str(exc)}
        return info
    if status.record is not None:
        info["generated"] = {
            "by": status.record["by"], "stale": status.stale, "edited": status.edited,
            "current": status.current, "reasons": list(status.reasons),
            "regenerate": regenerate_command(path, status.record)}
    return info


def chain(path):
    """``file_info`` of each file from the root of ``path``'s chain down to it."""
    infos, seen = [], set()
    current = os.path.abspath(path)
    while current is not None:
        if current in seen:
            raise ValueError("extends cycle at %s" % (current,))
        seen.add(current)
        info = file_info(current)
        infos.append(info)
        current = (os.path.normpath(os.path.join(os.path.dirname(current), info["extends"]))
                   if info["extends"] else None)
    return list(reversed(infos))


def descendants(path):
    """``{file: [the files beside it that extend it]}``, for the tree under ``path``.

    Only the YMF files in ``path``'s own directory are looked at, which is
    where ``pro weak`` and ``pro dis`` write. Archives extend nothing.
    """
    here = os.path.dirname(os.path.abspath(path))
    parents = {}
    for name in sorted(os.listdir(here)):
        if not name.endswith(".ymf") or name.endswith(".archive.ymf"):
            continue
        full = os.path.join(here, name)
        try:
            extends = _read(full).get("extends")
        except Exception:                 # not a YMF file this can read: not ours
            continue
        if extends:
            parents.setdefault(os.path.normpath(os.path.join(here, extends)), []).append(full)
    tree, todo = {}, [os.path.abspath(path)]
    while todo:
        node = todo.pop()
        tree[node] = parents.get(node, [])
        todo.extend(c for c in tree[node] if c not in tree)
    return tree


def stale(path):
    """The generated files in ``path``'s chain whose input changed since."""
    return [info for info in chain(path)
            if info["generated"] and (info["generated"].get("stale")
                                      or "error" in info["generated"])]
