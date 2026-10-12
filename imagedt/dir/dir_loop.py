"""Deterministic directory traversal preserving the original callbacks."""

import os
import re


class LoopFile:
    def __init__(
        self, root_dir, file_extend=None, short_exclude=None, long_exclude=None
    ):
        self.root_dir = os.fspath(root_dir)
        self.file_extend = tuple(file_extend or ())
        self.short_exclude = tuple(short_exclude or ())
        self.long_exclude = tuple(long_exclude or ())
        self._patterns = [re.compile(p) for p in self.short_exclude]

    def start(self, func):
        self.func = func
        return self.loop_file(self.root_dir)

    def loop_file(self, root_dir):
        results = []
        with os.scandir(root_dir) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if any(
                    p in entry.name or r.search(entry.name)
                    for p, r in zip(self.short_exclude, self._patterns, strict=True)
                ):
                    continue
                if any(entry.path.endswith(p) for p in self.long_exclude):
                    continue
                # Directory symlinks are excluded to prevent traversal cycles.
                if entry.is_dir(follow_symlinks=False):
                    results.extend(self.loop_file(entry.path))
                elif entry.is_file():
                    if (
                        self.file_extend
                        and os.path.splitext(entry.name)[1] not in self.file_extend
                    ):
                        continue
                    results.append(self.func(entry.path))
        return results


def loop(root, extensions=(".jpg", ".png")):
    return LoopFile(root, extensions).start(lambda path: path)


def get_dir_images(root="./", extensions=(".jpg", ".png")):
    return len(loop(root, extensions))
