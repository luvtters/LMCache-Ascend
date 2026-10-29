"""Experiment-only markers around the unchanged LMCache CLI entry."""
import atexit
import faulthandler
import os
import runpy
import signal
import sys
import time

if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    atexit.register(lambda: print('NIGHT_ATEXIT pid=%s time=%s' % (os.getpid(),time.time()),flush=True))
    sys.argv=['lmcache',*sys.argv[1:]]
    try:
        runpy.run_module('lmcache.cli.main',run_name='__main__')
    except SystemExit as error:
        print('NIGHT_CLI_SYSTEM_EXIT code=%r time=%s' % (error.code,time.time()),flush=True)
        raise
    else:
        print('NIGHT_CLI_RETURNED time=%s' % time.time(),flush=True)
