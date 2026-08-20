# Dataset generation tools

These scripts plan, certify, record, finalize, validate, and review deterministic
first-person simulator episodes. They are the orchestration layer used with a
compatible packaged simulator build; the Unreal Engine project, game assets,
and packaged executable are not included here.

The build may or may not be released in the future. There is no commitment or
release schedule. Publishing these scripts should not be interpreted as a
promise that the build or dataset will become available.

## Contents

- `Scripts/dataset_controller.py`: immutable Movement V1 planning and inventory
- `Scripts/v2_dataset_controller.py`: Trajectory/Throw V2 planning and review sets
- `Scripts/v2_mission_catalog.py`: deterministic construction of all 62 V2 mission types
- `Scripts/dataset_worker.py`: assignment execution against a supplied build
- `Scripts/finalize_production_dataset.py`: WebP/Parquet tar-shard finalization
- `Scripts/review_dataset.py`: shard validation and recorded-frame video rendering
- certification, replacement, format-comparison, and regression-test utilities
- `examples/`: small Windows batch/configuration examples for a packaged build
- `V2_MISSION_DESIGN.md`: mission geometry, allocation, and validation policy

## Requirements

Install the lightweight generator dependencies separately from the PyTorch
training environment:

```bash
python -m pip install -r data_generation/Scripts/requirements-production.txt
```

The controllers and most tests do not launch the simulator. Commands that
record or runtime-certify episodes accept the build explicitly through an
`--executable` argument.

## Typical flow

The exact command depends on the build and desired campaign size, but the
pipeline is:

1. Create an immutable V1 or V2 collection plan.
2. Verify the plan and, for guided missions, certify it against the build.
3. Run assignments with `dataset_worker.py`.
4. Finalize staging metadata and frames into tar shards.
5. Validate the shards and optionally render review videos.

Use each command's `--help` output for its full interface. For example:

```bash
python data_generation/Scripts/dataset_controller.py --help
python data_generation/Scripts/v2_dataset_controller.py --help
python data_generation/Scripts/dataset_worker.py --help
python data_generation/Scripts/review_dataset.py --help
```

Generated outputs are intentionally ignored by Git.
