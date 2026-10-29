"""Two workers, one latest-value slot per stage; GUI remains on the main thread."""
from collections import deque
from dataclasses import dataclass
import threading
import time


class LatestSlot:
    def __init__(self):
        self._condition = threading.Condition()
        self._version, self._value = 0, None
        self.closed = False

    def publish(self, value):
        with self._condition:
            if not self.closed:
                self._version += 1
                self._value = value
                self._condition.notify_all()

    def wait(self, after=0, timeout=None):
        with self._condition:
            self._condition.wait_for(lambda: self._version > after or self.closed, timeout)
            return (self._version, self._value) if self._version > after else None

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()


@dataclass
class Sample:
    value: object
    captured_at: float


@dataclass
class Result:
    sample: Sample
    value: object
    seconds: float
    finished_at: float


class LivePipeline:
    """Capture never waits for inference; inference drops superseded frames.

    infer(sample.value) runs on one worker that exclusively owns the estimator.
    Results retain their exact RGB-D input so a pose is never drawn on a newer
    image. Capture and inference exceptions are re-raised by the main thread.
    """
    def __init__(self, source, infer, initial, max_results=0, initial_time=None):
        self.source, self.infer = iter(source), infer
        self.initial = Sample(initial, initial_time or time.perf_counter())
        self.max_results = max_results
        self.frames, self.results = LatestSlot(), LatestSlot()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._errors = []
        self._capture_times = deque(maxlen=60)
        self._inference_times = deque(maxlen=60)
        self._captured, self._processed, self._dropped = 1, 0, 0
        self._first_track_at = None
        self._last_track_at = None
        self._threads = [threading.Thread(target=self._capture, name='RGB-D capture', daemon=True),
                         threading.Thread(target=self._process, name='Pose inference', daemon=True)]

    def start(self):
        for thread in self._threads:
            thread.start()
        return self

    def _failed(self, exc):
        with self._lock:
            self._errors.append(exc)
        self.stop()

    def _capture(self):
        try:
            for value in self.source:
                if self._stop.is_set():
                    break
                now = time.perf_counter()
                with self._lock:
                    self._captured += 1
                    self._capture_times.append(now)
                self.frames.publish(Sample(value, now))
        except Exception as exc:
            self._failed(exc)
        finally:
            close = getattr(self.source, 'close', None)
            if close:
                close()
            self.frames.close()

    def _process(self):
        sample, version = self.initial, 0
        try:
            while not self._stop.is_set():
                start = time.perf_counter()
                value = self.infer(sample.value)
                finish = time.perf_counter()
                with self._lock:
                    self._processed += 1
                    # Exclude slow initial registration from steady tracking rate.
                    if self._processed > 1:
                        self._inference_times.append(finish)
                        if self._first_track_at is None:
                            self._first_track_at = finish
                        self._last_track_at = finish
                self.results.publish(Result(sample, value, finish-start, finish))
                if self.max_results and self._processed >= self.max_results:
                    self._stop.set()
                    self.frames.close()
                    break
                item = self.frames.wait(version)
                if item is None:
                    break
                next_version, sample = item
                with self._lock:
                    self._dropped += next_version-version-1
                version = next_version
        except Exception as exc:
            self._failed(exc)
        finally:
            self.results.close()

    def stats(self):
        def rate(times):
            return (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else 0.
        with self._lock:
            overall = (self._processed-2)/(self._last_track_at-self._first_track_at) if self._processed > 2 else 0.
            return dict(captured=self._captured, processed=self._processed, dropped=self._dropped,
                        capture_fps=rate(self._capture_times), pose_fps=rate(self._inference_times),
                        pose_fps_overall=overall)

    def raise_if_failed(self):
        with self._lock:
            if self._errors:
                raise self._errors[0]

    def stop(self):
        self._stop.set()
        self.frames.close()
        self.results.close()

    def join(self):
        for thread in self._threads:
            thread.join()
