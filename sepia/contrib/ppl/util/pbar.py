"""
Basic progress bar.
"""

from datetime import timedelta, datetime
import time

def pbrange(*args, **kwargs):
    return pbar(range(*args), **kwargs)

class pbar:
    def __init__(self, iterable, min_interval=0.2, chunks=15, char="#"):
        self.iterator = iterable.__iter__()
        self.iters = iterable.__len__()
        self.min_interval = min_interval
        self.genesis = time.time()
        self.chunks = chunks
        self.char = char
        self.tic = self.genesis
        self.extra = None

    def parse_extra(self):
        if self.extra is None:
            return ""
        else:
            return " | " + " | ".join(f"{name}: {value}" for name, value in self.extra.items())

    def make_progress_bar(self):
        chunk = int(self.chunks * self.i / self.iters)
        return f"{chunk * self.char}{(self.chunks - chunk) * ' '}"

    def clean(self, x):
        return str(x).split(".")[0]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        toc = time.time()
        self.print(toc)
        self.i += 1
        if self.i > self.iters:
            print()
        return self.iterator.__next__()

    def print(self, toc):
        if self.i == 0:
            self.bar = (
                f"{self.clean(datetime.now()) }"
            )
            print(f"[{self.bar}]", end="\r", flush=True)
        elif toc - self.tic > self.min_interval or self.i == self.iters:
            elapsed = toc - self.genesis
            speed = f"{self.i / elapsed:.2f}" if self.i > 0 else "???"
            elapsed_dt = timedelta(seconds=elapsed)
            remaining_dt = elapsed_dt * (self.iters / self.i - 1)
            perc = int(100 * self.i / self.iters)
            self.bar = (
                f"{self.clean(datetime.now()) } | "
                f"{self.make_progress_bar()} | "
                f"{self.i}/{self.iters} ({perc}%) | "
                f"{self.clean(elapsed_dt)}<"
                f"{self.clean(remaining_dt)} | "
                f"{speed}it/s" + self.parse_extra()
            )
            print(f"[{self.bar}]", end="\r", flush=True)
            self.tic = toc

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

# Demo. TODO: Move to tests.
if __name__ == "__main__":
    n = 10000
    s = 1e-4

    def time_wo_pbar():
        tic = time.time()
        for _ in range(n):
            time.sleep(s)
        toc = time.time()
        print(f"Without progress bar: {n / (toc - tic):.2f}it/s")

    # The true speed.
    print(f"Expected it/s: {1/s:.2f}")

    # Print speed without the progress bar.
    # This is the fastest speed we can expect due to loop overhead.
    time_wo_pbar()

    # Basic usage.
    for i in pbar(range(n)):
        time.sleep(s)

    # Another basic usage.
    for i in pbrange(n):
        time.sleep(s)

    # Define pb.
    pb = pbar(range(n))
    for i in pb:
        time.sleep(s)

    # Context and add extra args.
    xs = range(0, 5 * n, 5)
    with pbar(xs) as pb:
        for i in pb:
            pb.extra = {f"{i}^2": i ** 2}
            time.sleep(s)

    # Context and add extra args, using prange.
    print("pbrange in context:")
    with pbrange(0, 5 * n, 5) as pb:
        for i in pb:
            pb.extra = {f"{i}^2": i ** 2}
            time.sleep(s)

    # Works with lists.
    with pbar(list(xs)) as pb:
        for x in pb:
            pb.extra = {f"{x}^2": x ** 2}
            time.sleep(s)

    # Works with numpy arrays. Note the speed difference.
    # import numpy as np
    # ys = np.array(xs)[:, None]
    # with pbar(ys) as pb:
    #     for y in pb:
    #         pb.extra = {f"{y}^2": y ** 2}
    #         time.sleep(s)

    # Messages are consistent with the iteration number.
    pb = pbrange(3)
    for i in pb:
        time.sleep(1)
        pb.extra = {"i+1": i + 1}

    # Converting to list behaves as expected.
    N = int(1e6)
    x = [i + 1 for i in pbrange(N)]
    assert x[0] == 1 and x[-1] == N
    assert list(range(5)) == list(pbrange(5))
