"""Progress-bar compatibility helpers."""

from __future__ import annotations

try:
    from tqdm import tqdm, trange
except ImportError:
    def trange(*args, **kwargs):
        kwargs.pop("desc", None)
        return range(*args, **kwargs)

    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def update(self, amount=1):
            return None

        def set_description(self, description):
            return None

        def close(self):
            return None
