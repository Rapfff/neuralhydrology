from neuralhydrology.nh_run import start_run
from pathlib import Path

start_run(config_file=Path("examples/07-Superflex/experiment.yml", gpu=-1))