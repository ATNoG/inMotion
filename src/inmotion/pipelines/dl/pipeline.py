"""DL pipeline orchestration."""

from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from inmotion.dl.config import DLConfig
from inmotion.dl.data_loader import DLDataLoader
from inmotion.dl.evaluation import (
    evaluate_model_on_test,
    metrics_to_extended_row,
    plot_training_curves,
    save_results_csv,
)
from inmotion.dl.models.ensemble import StackingEnsemble, VotingEnsemble
from inmotion.dl.models.moe import (
    MOE_COMBOS,
)
from inmotion.dl.optimization import (
    init_optuna_db,
)
from inmotion.dl.training import Trainer
from inmotion.pipelines.dl.builders import (
    _build_ds_variant,
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
from inmotion.pipelines.dl.workers import (
    _ds_base_worker,
    _hpo_worker,
    _meta_model_worker,
    _moe_mo_hpo_worker,
    _nas_worker,
    _single_model_worker,
    _softmoe_combo_worker,
)


def run(args: RunConfig) -> int:
    """Execute the DL pipeline.

    ``args`` carries the same field names the argparse namespace used to, so
    the body below is unchanged; the CLI builds it from typed options.
    """

    n_gpus = args.num_gpus if args.num_gpus > 0 else _available_gpus()
    devices = _build_device_pool(n_gpus)
    print(f"Device pool: {devices}  (parallel workers: {len(devices)})")

    config = DLConfig(
        data_path=args.data,
        seed=args.seed,
        num_epochs=args.epochs,
        n_trials=args.trials,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        use_metadata=not args.no_meta,
    )
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.models_dir is not None:
        config.models_dir = args.models_dir
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.plots_dir is not None:
        config.plots_dir = args.plots_dir
    if args.no_resume:
        config._force_retrain = True
    config.make_dirs()
    set_seed(config.seed)

    # ── Data (main process) ──────────────────────────────────────────────────
    data_loader = DLDataLoader(config)
    X, y = data_loader.load_and_preprocess()
    classes = data_loader.classes_
    print(f"Dataset: {X.shape}  classes: {classes}  seed: {config.seed}")

    X_train, X_test, y_train, y_test = data_loader.train_test_split(X, y)
    X_tr, X_val, y_tr, y_val = data_loader.train_test_split(X_train, y_train)

    all_results: list[dict[str, object]] = []
    curve_data: list[dict[str, object]] = []

    # Prevent DataLoader workers from forking after CUDA init -> SIGBUS (exit 135).
    # A RuntimeError means the start method is already set, which is fine.
    with contextlib.suppress(RuntimeError):
        torch.multiprocessing.set_start_method("spawn")
    ctx = mp.get_context("spawn")

    # ── Phase 1: Parallel single-model training ──────────────────────────────
    # Each model lands on its own GPU (round-robin). 2 GPUs → 2 models run simultaneously.
    print(f"\n[Phase 1] {len(SINGLE_MODEL_NAMES)} models × {len(devices)} GPU(s)")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _single_model_worker,
                name,
                devices[i % len(devices)],
                config,
                X_tr,
                y_tr,
                X_val,
                y_val,
                X_test,
                y_test,
                classes,
            ): name
            for i, name in enumerate(SINGLE_MODEL_NAMES)
        }
        with tqdm(total=len(futures), desc="Phase 1: single models", unit="model") as pbar:
            for future in as_completed(futures):
                row = future.result()
                val_mccs: list[float] = row.pop("val_mccs", [])
                all_results.append(row)
                curve_data.append({"name": str(row["model"]), "val_mccs": val_mccs})
                pbar.set_postfix_str(f"{row['model']} mcc={float(row['test_mcc']):.3f}")
                pbar.update(1)

    # ── Phase 1.5: Metadata-fusion models (noise + concurrent_noise_path) ────
    if config.use_metadata and not args.no_meta:
        print(
            f"\n[Phase 1.5] Metadata-fusion models ({len(META_MODEL_NAMES)}) — noise + path context"
        )
        meta_dl = DLDataLoader(config)
        X_m, meta_m, y_m = meta_dl.load_and_preprocess_with_meta()
        (
            X_m_train,
            X_m_test,
            meta_m_train,
            meta_m_test,
            y_m_train,
            y_m_test,
        ) = meta_dl.train_test_split_with_meta(X_m, meta_m, y_m)
        (
            X_m_tr,
            X_m_val,
            meta_m_tr,
            meta_m_val,
            y_m_tr,
            y_m_val,
        ) = meta_dl.train_test_split_with_meta(X_m_train, meta_m_train, y_m_train)

        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            meta_futures = {
                pool.submit(
                    _meta_model_worker,
                    name,
                    devices[i % len(devices)],
                    config,
                    X_m_tr,
                    meta_m_tr,
                    y_m_tr,
                    X_m_val,
                    meta_m_val,
                    y_m_val,
                    X_m_test,
                    meta_m_test,
                    y_m_test,
                    classes,
                ): name
                for i, name in enumerate(META_MODEL_NAMES)
            }
            with tqdm(total=len(meta_futures), desc="Phase 1.5: meta-fusion", unit="model") as pbar:
                for future in as_completed(meta_futures):
                    row = future.result()
                    mv: list[float] = row.pop("val_mccs", [])
                    all_results.append(row)
                    curve_data.append({"name": str(row["model"]), "val_mccs": mv})
                    pbar.set_postfix_str(f"{row['model']} mcc={float(row['test_mcc']):.3f}")
                    pbar.update(1)

    # ── Phase 2: Ensembles (main process — reload saved checkpoints) ─────────
    print("\n[Phase 2] Building ensembles…")
    trained_models: dict[str, nn.Module] = {}
    for name in SINGLE_MODEL_NAMES:
        m = _build_model_by_name(name, config)
        state = torch.load(
            config.models_dir / f"{name}_seed{config.seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        m.load_state_dict(state)
        trained_models[name] = m

    main_device = torch.device(devices[0])
    test_loader = data_loader.make_loader(X_test, y_test, shuffle=False)
    tr_loader = data_loader.make_loader(X_tr, y_tr, shuffle=True)
    val_loader = data_loader.make_loader(X_val, y_val, shuffle=False)

    ensemble_combos: dict[str, list[str]] = {
        "Ensemble_RNN_GRU": ["RNN", "GRU"],
        "Ensemble_GRU_LSTM": ["GRU", "LSTM"],
        "Ensemble_LSTM_CNN": ["LSTM", "CNN"],
        "Ensemble_CNN_TCN": ["CNN", "TCN"],
        # "Ensemble_TCN_Transformer": ["TCN", "Transformer"],  # Transformer commented out
        # "Ensemble_BiLSTM_LSTM": ["BiLSTM", "LSTM"],          # BiLSTM commented out
        "Ensemble_Mamba_LSTM": ["Mamba", "LSTM"],
        "Ensemble_RNN_GRU_LSTM": ["RNN", "GRU", "LSTM"],
        "Ensemble_All": SINGLE_MODEL_NAMES,
    }
    for ens_name, member_names in tqdm(
        ensemble_combos.items(), desc="Phase 2: ensembles", unit="combo"
    ):
        members = [trained_models[n] for n in member_names]
        ensemble = VotingEnsemble(members).to(main_device)
        metrics = evaluate_model_on_test(
            ensemble, test_loader, config, classes, ens_name, log_wandb=config.use_wandb
        )
        torch.save(
            ensemble.state_dict(),
            config.models_dir / f"{ens_name}_seed{config.seed}.pt",
        )
        all_results.append(
            metrics_to_extended_row(
                ens_name,
                "voting_ensemble",
                config.seed,
                str(main_device),
                metrics,
                extra={"members": "+".join(member_names)},
            )
        )

    print("  [Stacking_All] training meta-learner…")
    stacking = StackingEnsemble(
        list(trained_models.values()), config.num_classes, config.dropout
    ).to(main_device)
    stacking.freeze_base_models()
    stack_cfg = copy.copy(config)
    stack_cfg.device = str(main_device)
    stack_trainer = Trainer(stack_cfg, run_name="Stacking_All")
    stack_save = config.models_dir / f"Stacking_All_seed{config.seed}.pt"
    stack_result = stack_trainer.fit(stacking, tr_loader, val_loader, save_path=stack_save)
    metrics = evaluate_model_on_test(
        stacking, test_loader, config, classes, "Stacking_All", log_wandb=config.use_wandb
    )
    all_results.append(
        metrics_to_extended_row(
            "Stacking_All",
            "stacking_ensemble",
            config.seed,
            str(main_device),
            metrics,
            best_val_mcc=stack_result.best_val_mcc,
        )
    )
    curve_data.append({"name": "Stacking_All", "val_mccs": stack_result.val_mccs})
    print(f"  [Stacking_All] test_mcc={float(metrics['mcc']):.4f}")

    # ── Phase 3: Parallel Optuna HPO + NAS ───────────────────────────────────
    if not args.no_optuna:
        print(f"\n[Phase 3] HPO ({len(HPO_TYPES)} studies) + NAS across {len(devices)} GPU(s)…")
        # Pre-init DB schema + studies in main process → no spawn race
        all_study_names = [f"{config.optuna_study_prefix}_{t}" for t in HPO_TYPES] + [
            f"{config.optuna_study_prefix}_nas"
        ]
        init_optuna_db(config.optuna_storage, all_study_names)
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            hpo_futures = {
                pool.submit(
                    _hpo_worker,
                    model_type,
                    devices[i % len(devices)],
                    config,
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): model_type
                for i, model_type in enumerate(HPO_TYPES)
            }
            with tqdm(total=len(hpo_futures), desc="Phase 3: HPO studies", unit="study") as pbar:
                for future in as_completed(hpo_futures):
                    res = future.result()
                    hpo_val_mccs: list[float] = list(res.pop("val_mccs", []))  # type: ignore[arg-type]
                    per_class = res.pop("per_class", {})
                    cm = res.pop("confusion_matrix", None)
                    row = {
                        "model": res["model"],
                        "type": res["type"],
                        "seed": res["seed"],
                        "device": res["device"],
                        "test_mcc": res["test_mcc"],
                        "test_acc": res["test_acc"],
                        "best_val_mcc": res["best_val_mcc"],
                        "best_epoch": res.get("best_epoch", 0),
                        "train_time_s": res.get("train_time_s", 0.0),
                        "precision_macro": res.get("precision_macro", 0.0),
                        "recall_macro": res.get("recall_macro", 0.0),
                        "f1_macro": res.get("f1_macro", 0.0),
                        "precision_weighted": res.get("precision_weighted", 0.0),
                        "recall_weighted": res.get("recall_weighted", 0.0),
                        "f1_weighted": res.get("f1_weighted", 0.0),
                    }
                    # Per-class metrics
                    for cls_name, cls_metrics in per_class.items():
                        row[f"{cls_name}_precision"] = cls_metrics.get("precision", 0)
                        row[f"{cls_name}_recall"] = cls_metrics.get("recall", 0)
                        row[f"{cls_name}_f1"] = cls_metrics.get("f1", 0)
                        row[f"{cls_name}_support"] = cls_metrics.get("support", 0)
                    # Confusion matrix
                    if cm is not None:
                        row["Confusion_Matrix"] = json.dumps(cm.tolist())
                        classes_list = list(per_class.keys()) or classes
                        for i, true_cls in enumerate(classes_list):
                            for j, pred_cls in enumerate(classes_list):
                                row[f"CM_{true_cls}_pred_{pred_cls}"] = int(cm[i, j])
                    all_results.append(row)
                    curve_data.append({"name": str(res["model"]), "val_mccs": hpo_val_mccs})
                    pbar.set_postfix_str(
                        f"HPO {res['model_type']} mcc={float(res['test_mcc']):.3f}"
                    )
                    pbar.update(1)

        print("  [NAS] architecture search…")
        nas_row = _nas_worker(
            devices[0],
            config,
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            y_test,
            classes,
        )
        nas_val_mccs: list[float] = nas_row.pop("val_mccs", [])  # type: ignore[assignment]
        all_results.append(nas_row)
        curve_data.append({"name": "OptunaNet_NAS", "val_mccs": nas_val_mccs})
        nas_mcc = float(nas_row.get("test_mcc", 0))  # type: ignore[arg-type]
        nas_arch = nas_row.get("best_arch", "{}")
        print(f"  [NAS] {nas_row.get('device', 'cpu')}  arch={nas_arch}  test_mcc={nas_mcc:.4f}")

    # ── Phase 4: Mixture-of-Experts — end-to-end SoftMoE with multi-objective HPO ─
    if not args.no_moe:
        n_combos = len(MOE_COMBOS)
        print(
            f"\n[Phase 4a] SoftMoE multi-objective HPO (NSGA-II) — "
            f"{n_combos} combos across {len(devices)} GPU(s)…"
        )
        # Multi-objective studies use directions=[] so cannot use init_optuna_db directly;
        # they are created inside the worker / study call (load_if_exists=True).

        # Phase 4a: run multi-objective HPO for each combo in parallel across GPUs
        combo_best_params: dict[str, dict] = {}
        if not args.no_optuna:
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
                mo_futures = {
                    pool.submit(
                        _moe_mo_hpo_worker,
                        combo_name,
                        arch_list,
                        devices[i % len(devices)],
                        copy.copy(config),
                        X_tr,
                        y_tr,
                        X_val,
                        y_val,
                    ): combo_name
                    for i, (combo_name, arch_list) in enumerate(MOE_COMBOS.items())
                }
                with tqdm(total=len(mo_futures), desc="Phase 4a: MoE HPO", unit="combo") as pbar:
                    for future in as_completed(mo_futures):
                        res = future.result()
                        combo_best_params[str(res["combo_name"])] = dict(res["best_params"])  # type: ignore[arg-type]
                        pbar.set_postfix_str(
                            f"{res['combo_name']} mcc={float(res['pareto_best_mcc']):.3f}"
                        )
                        pbar.update(1)
        else:
            combo_best_params = {name: {} for name in MOE_COMBOS}

        # Phase 4b: train each combo end-to-end with best params
        print(
            f"\n[Phase 4b] SoftMoE e2e training — {n_combos} combos across {len(devices)} GPU(s)…"
        )
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            train_futures = {
                pool.submit(
                    _softmoe_combo_worker,
                    combo_name,
                    arch_list,
                    devices[i % len(devices)],
                    copy.copy(config),
                    combo_best_params.get(combo_name, {}),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): combo_name
                for i, (combo_name, arch_list) in enumerate(MOE_COMBOS.items())
            }
            with tqdm(
                total=len(train_futures), desc="Phase 4b: SoftMoE training", unit="combo"
            ) as pbar:
                for future in as_completed(train_futures):
                    res = future.result()
                    moe_val_mccs: list[float] = list(res.pop("val_mccs", []))  # type: ignore[arg-type]
                    all_results.append(res)
                    curve_data.append({"name": str(res["model"]), "val_mccs": moe_val_mccs})
                    pbar.set_postfix_str(f"{res['model']} mcc={float(res['test_mcc']):.3f}")
                    pbar.update(1)

    # ── Phase 5: Deep Stacking ─────────────────────────────────────────────────
    if not args.no_deepstack:
        n_base = len(DS_BASE_NAMES)
        print(f"\n[Phase 5] DeepStack — {n_base} base models across {len(devices)} GPU(s)…")
        ds_config = copy.copy(config)
        dl_ds = DLDataLoader(ds_config)
        tr_loader_ds = dl_ds.make_loader(X_tr, y_tr, shuffle=True)
        val_loader_ds = dl_ds.make_loader(X_val, y_val, shuffle=False)
        # The DeepStack test split is loaded where it is used, in stage 2.

        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            ds_futures = {
                pool.submit(
                    _ds_base_worker,
                    name,
                    devices[i % len(devices)],
                    copy.copy(config),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    classes,
                ): name
                for i, name in enumerate(DS_BASE_NAMES)
            }
            ds_base_results: list[dict[str, object]] = []
            with tqdm(total=len(ds_futures), desc="Phase 5: DS base models", unit="model") as pbar:
                for future in as_completed(ds_futures):
                    res = future.result()
                    ds_base_results.append(res)
                    all_results.append({k: v for k, v in res.items() if k != "save_path"})
                    pbar.set_postfix_str(f"{res['model']} mcc={float(res['test_mcc']):.3f}")
                    pbar.update(1)

        ds_base_results.sort(key=lambda d: str(d["model"]))

        # ── Reload base models and extract concat logits (N, n_base * num_classes) ─
        dev0 = torch.device(devices[0])
        base_models_loaded: list[nn.Module] = []
        for info in sorted(ds_base_results, key=lambda d: str(d["model"])):
            _raw = torch.load(str(info["save_path"]), map_location="cpu", weights_only=False)
            if isinstance(_raw, nn.Module):
                m = _raw  # new format: full model saved
            else:
                # legacy format: state dict only — rebuild from current arch definition
                m = _build_ds_variant(str(info["model"]), config)
                m.load_state_dict(_raw)
            base_models_loaded.append(m)

        def _concat_logits(loader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]") -> np.ndarray:
            return np.concatenate(
                [_extract_logits(m, loader, dev0) for m in base_models_loaded], axis=-1
            )

        # Only the train and validation folds feed the level-2 nets and the meta
        # model; the test fold is evaluated at the end of stage 2. Base-model
        # logits for the test split were previously computed here and discarded.
        base_feat_tr = _concat_logits(tr_loader_ds)  # (N, n_base * C)
        base_feat_val = _concat_logits(val_loader_ds)

        # ── Stage 1: Freeze base models, train Level2 nets (full multiclass) ──
        # Level2Net now outputs C classes (not binary), preserving full inter-class signal.
        from inmotion.dl.models.deep_stack import DeepMetaMLP as _DeepMetaMLP
        from inmotion.dl.models.deep_stack import Level2Net as _Level2Net

        n_base_feats = base_feat_tr.shape[1]  # n_base * C
        l2_config = copy.copy(config)
        l2_dl = DLDataLoader(l2_config)
        level2_nets: list[nn.Module] = []

        print("  [DeepStack stage-1] training Level2 nets (multiclass)…")
        for cls_name in tqdm(classes, desc="Phase 5: L2 nets", unit="class"):
            l2_tr = l2_dl.make_loader(base_feat_tr, y_tr, shuffle=True)
            l2_val = l2_dl.make_loader(base_feat_val, y_val, shuffle=False)
            l2_net: nn.Module = _Level2Net(n_base_feats, config.num_classes, dropout=config.dropout)
            Trainer(l2_config).fit(l2_net, l2_tr, l2_val)
            save_path_l2 = config.models_dir / f"DS_L2_{cls_name}_seed{config.seed}.pt"
            torch.save(l2_net.state_dict(), save_path_l2)
            level2_nets.append(l2_net)

        # ── Extract Level2 features: concat softmax of each L2 net → (N, n_l2 * C) ──
        def _l2_feats(feats: np.ndarray) -> np.ndarray:
            feat_t = torch.tensor(feats, dtype=torch.float32).to(dev0)
            parts = []
            for l2 in level2_nets:
                l2.eval().to(dev0)
                with torch.no_grad():
                    parts.append(torch.softmax(l2(feat_t), dim=-1).cpu().numpy())
            return np.concatenate(parts, axis=-1)  # (N, n_l2 * C)

        l2_feat_tr = _l2_feats(base_feat_tr)
        l2_feat_val = _l2_feats(base_feat_val)
        # NOTE: stage 1 (DeepMetaMLP) is trained but never evaluated on the test
        # split; stage 2 does the reported evaluation. Computing the test L2
        # features here would be discarded, so it is not done.

        # ── Train DeepMetaMLP on L2 features ──────────────────────────────────
        print("  [DeepStack stage-1] training DeepMetaMLP…")
        meta_ds_config = copy.copy(config)
        meta_ds_dl = DLDataLoader(meta_ds_config)
        ds_meta_tr = meta_ds_dl.make_loader(l2_feat_tr, y_tr, shuffle=True)
        ds_meta_val = meta_ds_dl.make_loader(l2_feat_val, y_val, shuffle=False)

        n_l2_feats = l2_feat_tr.shape[1]  # n_l2 * C
        ds_meta: nn.Module = _DeepMetaMLP(n_l2_feats, config.num_classes, config.dropout)
        ds_meta_save = config.models_dir / f"DS_meta_seed{config.seed}.pt"
        stage1_trainer = Trainer(meta_ds_config, run_name="DeepStackMeta_S1")
        # Stage 1 trains and saves the DeepMetaMLP; its result object is unused.
        stage1_trainer.fit(ds_meta, ds_meta_tr, ds_meta_val, save_path=ds_meta_save)

        # ── Stage 2: joint end-to-end fine-tuning (DeepStackEnsemble) ─────────
        # Reassemble the full pipeline into a single nn.Module for joint fine-tune.
        # Use a low LR (10× smaller) to avoid overwriting learned representations.
        print("  [DeepStack stage-2] joint end-to-end fine-tuning…")
        from inmotion.dl.models.deep_stack import DeepStackEnsemble as _DeepStackEnsemble

        # Rebuild Level2 nets with loaded state (on CPU initially)
        l2_nets_loaded: list[nn.Module] = []
        for cls_name in classes:
            l2_r = _Level2Net(n_base_feats, config.num_classes, dropout=config.dropout)
            l2_r.load_state_dict(
                torch.load(
                    config.models_dir / f"DS_L2_{cls_name}_seed{config.seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            l2_nets_loaded.append(l2_r)

        deep_stack = _DeepStackEnsemble(
            base_models=base_models_loaded,
            level2_nets=l2_nets_loaded,
            num_classes=config.num_classes,
            dropout=config.dropout,
        ).to(dev0)

        # Transfer meta weights from stage-1
        deep_stack.meta.load_state_dict(ds_meta.state_dict())

        # Freeze base models for stage 2 (they are already well-trained)
        deep_stack.freeze_base_models()

        ft_config = copy.copy(config)
        ft_config.device = str(dev0)
        ft_config.learning_rate = config.learning_rate * 0.1  # smaller LR for fine-tuning
        ft_config.num_epochs = min(50, config.num_epochs // 2)
        ft_config.patience = 12

        ft_tr = dl_ds.make_loader(X_tr, y_tr, shuffle=True)
        ft_val = dl_ds.make_loader(X_val, y_val, shuffle=False)
        ft_test = dl_ds.make_loader(X_test, y_test, shuffle=False)

        ft_trainer = Trainer(ft_config, run_name="DeepStackEnsemble")
        ft_save = config.models_dir / f"DeepStackEnsemble_seed{config.seed}.pt"
        ft_result = ft_trainer.fit(deep_stack, ft_tr, ft_val, save_path=ft_save)

        ds_metrics = evaluate_model_on_test(
            deep_stack,
            ft_test,
            ft_config,
            classes,
            "DeepStackEnsemble",
            log_wandb=config.use_wandb,
        )
        all_results.append(
            metrics_to_extended_row(
                "DeepStackEnsemble",
                "deep_stack",
                config.seed,
                str(dev0),
                ds_metrics,
                best_val_mcc=ft_result.best_val_mcc,
                extra={"val_mccs": ft_result.val_mccs},
            )
        )
        curve_data.append({"name": "DeepStackEnsemble", "val_mccs": ft_result.val_mccs})
        ds_mcc = float(ds_metrics["mcc"])
        ds_acc = float(ds_metrics["accuracy"])
        print(f"  [DeepStack] test_mcc={ds_mcc:.4f}  acc={ds_acc:.4f}")

    # ── Artefacts ─────────────────────────────────────────────────────────────
    results_csv = config.results_dir / f"dl_results_seed{config.seed}.csv"
    save_results_csv(all_results, results_csv)
    plot_training_curves(curve_data, config.plots_dir / "training_curves.png")

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Type':<22} {'MCC':>8} {'Acc':>8}")
    print("-" * 70)
    for row in sorted(all_results, key=lambda r: float(str(r.get("test_mcc", 0))), reverse=True):
        print(
            f"{str(row['model']):<30} {str(row['type']):<22} "
            f"{float(str(row.get('test_mcc', 0))):>8.4f} "
            f"{float(str(row.get('test_acc', 0))):>8.4f}"
        )
    print("=" * 70)
    print(f"\nResults → {results_csv}")
    return 0
