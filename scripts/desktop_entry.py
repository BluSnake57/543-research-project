import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if '--startup-check' in sys.argv:
        from sentinel_x.diagnostics import startup_check
        raise SystemExit(startup_check())
    from sentinel_x.desktop import main
    raise SystemExit(main())
