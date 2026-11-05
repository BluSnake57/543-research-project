import multiprocessing
from sentinel_x.trainer import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
