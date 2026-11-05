"""Minimal independent guardian. The SSH password stays in anonymous stdin only."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parent/'088-local-runtime-move/private/serving-105'
NATIVE_EXITS = {0xC0000005, 0xC0000374, 0xC0000409}


def restart_allowed(exit_code, exits, now):
    while exits and now-exits[0] >= 600:
        exits.popleft()
    if exit_code & 0xffffffff not in NATIVE_EXITS or len(exits) >= 3:
        return False
    exits.append(now)
    return True


def log(**values):
    row = dict(utc_epoch=time.time(), pid=os.getpid(), **values)
    with (PRIVATE/'watchdog.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')


def supervise(request, spawn, *, clock=time.monotonic, pause=time.sleep, emit=log):
    exits = collections.deque()
    while True:
        child = spawn(request)
        emit(event='supervisor_started', supervisor_pid=child.pid, run_id=request['run_id'])
        code = child.wait()
        allowed = restart_allowed(code, exits, clock())
        emit(event='supervisor_exited', supervisor_pid=child.pid, exit_code=code,
             native_exit=code & 0xffffffff in NATIVE_EXITS, restarting=allowed)
        if not allowed:
            request.clear()
            return 2
        pause(2)


def spawn(request):
    environment = os.environ.copy()
    environment['PYTHONFAULTHANDLER'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    with (PRIVATE/'supervisor.jsonl').open('ab', buffering=0) as stream:
        child = subprocess.Popen([sys.executable, '-B', str(HERE/'operations.py'), '--background'],
            stdin=subprocess.PIPE, stdout=stream, stderr=stream, cwd=HERE, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        child.stdin.write((json.dumps(request)+'\n').encode())
        child.stdin.close()
    return child


def main():
    request = json.loads(sys.stdin.buffer.readline(8192))
    return supervise(request, spawn)


if __name__ == '__main__':
    raise SystemExit(main())
