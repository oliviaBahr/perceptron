from typing import get_args

import comet_ml

from experiment_utils import ARCH_TYPE, DATA_DIRS, ENSEMBLE_TYPE, run_experiment

comet_ml.login()

NUM_RUNS = 100
NUM_ITERATIONS = 100  # num epochs for single models, num learners for ensembles
NUM_EPOCHS_PER_LEARNER = 1
TRAINING_SIZE = 0.5

for dirpath in DATA_DIRS:
    for arch in get_args(ARCH_TYPE):
        # Skip the MLP for baselines, let's make that a separate study
        if "mlp" in arch:
            continue

        for ensemble_type in get_args(ENSEMBLE_TYPE):
            print(f"Training {arch} - {ensemble_type}")

            run_experiment(
                project_name="baselines_v3",
                n_runs=NUM_RUNS,
                dirpath=dirpath,
                arch=arch,
                ensemble_type=ensemble_type,
                max_epochs_per_learner=(NUM_ITERATIONS if ensemble_type == "none" else NUM_EPOCHS_PER_LEARNER),
                num_learners=NUM_ITERATIONS,
                training_size=TRAINING_SIZE,
                learner_kwargs={},
            )
