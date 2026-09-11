"""DL pipeline: supervised baselines, HPO/NAS, MoE and DeepStack.

Split from a single 1,700-line module into:

``config``    model lists and the typed :class:`RunConfig`
``devices``   GPU pool selection and seeding
``builders``  model construction, including from saved HPO parameters
``workers``   spawn-safe worker functions used by ``ProcessPoolExecutor``
``pipeline``  :func:`run`, the orchestration
"""

from inmotion.pipelines.dl.builders import (
    _build_ds_variant,
    _build_hpo_best_model,
    _build_model_by_name,
    _extract_logits,
)
from inmotion.pipelines.dl.config import (
    DS_BASE_NAMES,
    HPO_TYPES,
    META_MODEL_NAMES,
    SINGLE_MODEL_NAMES,
    RunConfig,
)
from inmotion.pipelines.dl.devices import _available_gpus, _build_device_pool, set_seed
from inmotion.pipelines.dl.pipeline import run

__all__ = [
    "DS_BASE_NAMES",
    "HPO_TYPES",
    "META_MODEL_NAMES",
    "SINGLE_MODEL_NAMES",
    "RunConfig",
    "_available_gpus",
    "_build_device_pool",
    "_build_ds_variant",
    "_build_hpo_best_model",
    "_build_model_by_name",
    "_extract_logits",
    "run",
    "set_seed",
]
