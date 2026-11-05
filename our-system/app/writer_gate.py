"""Original reentrant FIFO/priority policy, with content-free slow-lock records."""
from collections import deque
import json
import sys
import threading
import time


class WriterGate:
    def __init__(self):
        self._condition = threading.Condition()
        self._owner = None
        self._depth = 0
        self._normal = deque()
        self._priority = deque()
        self._started = 0
        self._waited = 0
        self._operation = []

    @staticmethod
    def operation():
        frames = []
        frame = sys._getframe(2)
        for _ in range(12):
            if frame is None:
                break
            module = frame.f_globals.get('__name__', '')
            if module.startswith('app.') and module != 'app.writer_gate':
                frames.append({'module':module, 'function':frame.f_code.co_name,
                               'line':frame.f_lineno})
            frame = frame.f_back
        return frames[:6]

    def acquire(self, blocking=True, timeout=-1, *, priority=False):
        ident = threading.get_ident()
        entered = time.monotonic()
        deadline = None if timeout < 0 else entered + timeout
        token = object()
        with self._condition:
            if self._owner == ident:
                self._depth += 1
                return True
            queue = self._priority if priority else self._normal
            queue.append(token)
            try:
                while True:
                    head = self._priority if self._priority else self._normal
                    if self._owner is None and head[0] is token:
                        queue.popleft()
                        self._owner, self._depth = ident, 1
                        self._started = time.monotonic()
                        self._waited = self._started-entered
                        self._operation = self.operation()
                        return True
                    remaining = None if deadline is None else deadline-time.monotonic()
                    if not blocking or (remaining is not None and remaining <= 0):
                        return False
                    self._condition.wait(remaining)
            finally:
                if token in queue:
                    queue.remove(token)
                    self._condition.notify_all()

    def release(self):
        report = None
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError('writer gate released by non-owner')
            self._depth -= 1
            if not self._depth:
                elapsed = time.monotonic()-self._started
                if elapsed >= .25 or self._waited >= 1:
                    report = dict(held_ms=round(elapsed*1000, 3),
                                  waited_ms=round(self._waited*1000, 3),
                                  queued_normal=len(self._normal), queued_priority=len(self._priority),
                                  operation=self._operation)
                self._owner = None
                self._condition.notify_all()
        if report is not None:
            try:
                print('AML_LOCK_092 '+json.dumps(report), flush=True)
            except (OSError, ValueError):
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
