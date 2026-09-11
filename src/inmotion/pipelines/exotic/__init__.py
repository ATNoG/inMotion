"""World-model (self-supervised) pipeline.

Split from a single 1,950-line module into modules with one job each:

``config``     the typed :class:`RunConfig` (54 options)
``devices``    device selection and seeding
``data``       pretraining and supervised data loading
``builders``   the model family and its dispatch table
``training``   stage-1 pretraining, linear probing, stage-2 fine-tuning
``hpo``        shared Optuna helpers and per-trial logging
``hpo_jepa``   the JEPA search (also holds the Pareto machinery)
``hpo_sigreg`` the SIGReg search
``hpo_mamba3`` the Mamba-3 search
``pipeline``   :func:`run`, the orchestration
"""

from inmotion.pipelines.exotic.builders import MODEL_BUILDERS, _build_mamba3
from inmotion.pipelines.exotic.config import RunConfig
from inmotion.pipelines.exotic.data import load_pretrain_data, load_supervised_data
from inmotion.pipelines.exotic.devices import resolve_device, set_seed
from inmotion.pipelines.exotic.hpo import _run_hpo
from inmotion.pipelines.exotic.pipeline import run
from inmotion.pipelines.exotic.training import finetune_jepa, pretrain_jepa

__all__ = [
    "MODEL_BUILDERS",
    "RunConfig",
    "_build_mamba3",
    "_run_hpo",
    "finetune_jepa",
    "load_pretrain_data",
    "load_supervised_data",
    "pretrain_jepa",
    "resolve_device",
    "run",
    "set_seed",
]
