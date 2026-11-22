"""Command line: python -m dh.cli <stage>.

Stages, in order: download, build, validate, [validate_resid], freeze, final, [final_resid], test, report. Each
writes its outputs to files, so a stage can be rerun alone. `test` runs once: it refuses to run again unless --force
(which is logged). validate_resid / final_resid are the residual hedgers added for test run 3.
"""

import argparse
import logging
import sys

from dh import config


def setup_logging(stage: str) -> None:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOGS / f"{stage}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="dh")
    p.add_argument("stage", choices=["download", "build", "validate", "validate_resid", "freeze", "final", "final_resid",
                                     "test", "report"])
    p.add_argument("--force", action="store_true", help="test stage: allow a second (logged) test run")
    p.add_argument("--only", default=None, help="validate/final(_resid): comma-separated generators to (re)train")
    a = p.parse_args(argv)
    setup_logging(a.stage)
    if a.stage == "download":
        from dh import download
        download.run()
    elif a.stage == "build":
        from dh import dataset
        dataset.build_all()
    elif a.stage == "validate":
        from dh import pipeline
        pipeline.validate(only=a.only.split(",") if a.only else None)
    elif a.stage == "validate_resid":
        from dh import pipeline
        pipeline.validate_resid(only=a.only.split(",") if a.only else None)
    elif a.stage == "final_resid":
        from dh import pipeline
        pipeline.final_resid(only=a.only.split(",") if a.only else None)
    elif a.stage == "freeze":
        from dh import pipeline
        pipeline.freeze()
    elif a.stage == "final":
        from dh import pipeline
        pipeline.final(only=a.only.split(",") if a.only else None)
    elif a.stage == "test":
        from dh import pipeline
        pipeline.test(force=a.force)
    elif a.stage == "report":
        from dh import report
        report.run()


if __name__ == "__main__":
    main()
