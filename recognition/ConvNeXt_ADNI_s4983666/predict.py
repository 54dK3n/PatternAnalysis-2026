"""Reload a checkpoint and evaluate its development validation fold.

Use ``python3 predict.py --help`` for command-line options.
"""

from engine.prediction import main


if __name__ == "__main__":
    raise SystemExit(main())
