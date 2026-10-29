"""Opt-in wall-clock stages, with device synchronization only while profiling."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import time

_active = ContextVar('foundationpose_stage_profile', default=None)


class StageProfile:
    def __init__(self, synchronize=None, clock=time.perf_counter):
        self.synchronize = synchronize or (lambda: None)
        self.clock = clock
        self._stack = []
        self._stages = {}

    def __enter__(self):
        self._token = _active.set(self)
        return self

    def __exit__(self, *exc):
        _active.reset(self._token)

    @contextmanager
    def span(self, name):
        self.synchronize()
        path = '/'.join([frame['name'] for frame in self._stack]+[name])
        frame = dict(name=name, child_seconds=0.)
        start = self.clock()
        self._stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            try:
                self.synchronize()
            finally:
                elapsed = self.clock()-start
                self._stack.pop()
                record = self._stages.setdefault(path, dict(calls=0, failed_calls=0,
                                                            inclusive_seconds=0., exclusive_seconds=0.))
                record['calls'] += 1
                record['failed_calls'] += int(failed)
                record['inclusive_seconds'] += elapsed
                record['exclusive_seconds'] += max(0., elapsed-frame['child_seconds'])
                if self._stack:
                    self._stack[-1]['child_seconds'] += elapsed

    def report(self):
        return dict(protocol='Wall clock; synchronize before/after each active stage. '
                             'Nested exclusive times exclude child stages. Profiling can change scheduling; '
                             'use uninstrumented A/B totals for speed claims.',
                    stages={key: dict(value) for key, value in self._stages.items()})

    def save(self, path):
        Path(path).write_text(json.dumps(self.report(), indent=2))


def stage(name):
    profile = _active.get()
    return profile.span(name) if profile is not None else nullcontext()


def profiled(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with stage(name):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def device_synchronize(device):
    import torch
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
