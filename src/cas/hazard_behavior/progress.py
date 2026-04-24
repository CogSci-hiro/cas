"""Progress helpers for behavioural hazard analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from tqdm.auto import tqdm


def progress_iterable(
    iterable: Iterable,
    *,
    total: int | None,
    description: str,
    enabled: bool = True,
) -> Iterator:
    """Wrap an iterable in a tqdm progress bar when enabled."""

    if not enabled:
        yield from iterable
        return
    yield from tqdm(
        iterable,
        total=total,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )
