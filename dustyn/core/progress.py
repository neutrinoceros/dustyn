from typing import Optional, Sequence

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    Text,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dustyn._typing import FloatLike


class CompletionColumn(ProgressColumn):
    """Renders the current number of iterations completed."""

    def render(self, task: Task) -> Text:
        return Text(f"{task.completed} it", style="progress.percentage")


class CompletionSpeedColumn(ProgressColumn):
    """Renders human readable completion speed."""

    # This is based off rich.progress.TransferSpeedColumn

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        data_speed = int(speed)
        return Text(f"{data_speed} it/s", style="progress.data.speed")


class RichProgressBar:
    """A thin wrapper around rich.progress.Progress"""

    def __init__(
        self,
        title: str,
        total: Optional[FloatLike] = None,
        columns: Optional[Sequence[ProgressColumn]] = None,
    ):
        if columns is None:
            columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ]
            if total is None:
                columns += [
                    CompletionColumn(),
                    TextColumn("[progress.data.speed]@"),
                    CompletionSpeedColumn(),
                ]
                # this is a workaround. As of rich 9.13.0, some of the columns
                # we're using here require a non-None value in `task.total` to render
                total = np.inf
            else:
                columns += [
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                ]
            columns += [TimeElapsedColumn()]
        self._pbar = Progress(*columns)

        self._pbar.start()
        self.task = self._pbar.add_task(title, total=total)

    def update(
        self,
        *,
        completed: Optional[FloatLike] = None,
        advance: Optional[FloatLike] = None,
        **kwargs,
    ) -> None:
        # We only expose the relevant subset of keyword arguments
        # supported by `rich.progress.Progress.update`
        # note that they should NOT be combined (use either `completed` or `advance` but not both)
        self._pbar.update(self.task, completed=completed, advance=advance, **kwargs)

    def refresh(self):
        self._pbar.refresh()

    def stop(self):
        self._pbar.stop()


def get_pbar(
    title: str, total: Optional[FloatLike] = None, start: bool = True
) -> Progress:
    return RichProgressBar(title, total)


class DummyProgressBar:
    # mock rich.progress.Progress's interface
    def update(
        self,
        *,
        completed: Optional[FloatLike] = None,
        advance: Optional[FloatLike] = None,
        **kwargs,
    ) -> None:
        pass

    def refresh(self):
        pass

    def stop(self):
        pass


if __name__ == "__main__":
    from time import sleep

    pbar = get_pbar(title="progress with advance...", total=70.0)
    for _ in range(70):
        sleep(0.02)
        pbar.update(advance=1)
    pbar.stop()

    pbar = get_pbar(title="progress with completed...", total=70.0)
    for i in range(70):
        sleep(0.02)
        pbar.update(completed=i + 1)
    pbar.stop()
