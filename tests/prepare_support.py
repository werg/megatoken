"""Crash injection for the preparation resume invariants."""

from __future__ import annotations

from megatoken.data import prepare as prepare_module


def interrupt_after_rows(monkeypatch, count: int) -> None:
    """Stop preparation as the source yields row ``count + 1``, as a crash would.

    Rows 1..``count`` have been fully processed; the run raises InterruptedError
    before reading another.
    """
    real = prepare_module.iter_source_rows

    def interrupted(source, *, start_row=0):
        seen = 0
        for item in real(source, start_row=start_row):
            if seen >= count:
                raise InterruptedError("Injected preparation interruption")
            seen += 1
            yield item

    monkeypatch.setattr(prepare_module, "iter_source_rows", interrupted)
