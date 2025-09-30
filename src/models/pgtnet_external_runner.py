"""
PGTNetExternalRunner

Runs the PGTNet/GraphGPS pipeline as an external process for a given dataset/fold/seed.
This runner avoids importing third-party code and keeps their environment isolated.

Generated artifacts per fold:
- outputs/pgtnet/<dataset>/fold_<k>/run_manifest.json
- outputs/pgtnet/<dataset>/fold_<k>/stdout.log, stderr.log
- outputs/pgtnet/<dataset>/fold_<k>/predictions_test.csv (if auto-collected)
- outputs/pgtnet/<dataset>/fold_<k>/metrics.json (optionally computed by the task)

Notes:
- We only construct and execute subprocess commands. Conversion/training/inference
  scripts remain the responsibility of PGTNet/GraphGPS repositories.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf
import yaml


@dataclass
class ExternalCallResult:
    cmd: List[str]
    returncode: int
    seconds: float
    stdout_path: Path
    stderr_path: Path


class PGTNetExternalRunner:
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset = dataset_name
        self.config = config or {}
        # Merge base model config with optional nested model.pgtnet overrides.
        base_model = dict(self.config.get("model", {}) or {})
        nested_over = dict((base_model.get("pgtnet", {}) or {}))
        # Remove nested key to avoid accidental passthrough later
        if "pgtnet" in base_model:
            try:
                base_model = {k: v for k, v in base_model.items() if k != "pgtnet"}
            except Exception:
                pass
        # Load default pgtnet model config from repo if required keys are missing (Hydra group not used)
        defaults: Dict[str, Any] = {}
        try:
            project_root = Path(__file__).resolve().parents[2]
            default_cfg_path = project_root / "configs" / "model" / "pgtnet.yaml"
            if default_cfg_path.exists():
                defaults = OmegaConf.to_container(OmegaConf.load(str(default_cfg_path)), resolve=True) or {}
        except Exception:
            defaults = {}
        # Deep-merge: nested dicts are updated recursively; scalars in overrides replace base.
        def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(a)
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        # Precedence: defaults <- base_model <- nested_overrides
        merged = _deep_merge(defaults, base_model)
        self.model_cfg = _deep_merge(merged, nested_over)
        # Backward compatibility: if nothing found, fall back to base model as-is
        if not self.model_cfg:
            self.model_cfg = self.config.get("model", {})
        self.outputs_root = Path(self.model_cfg.get("outputs_dir", "outputs/pgtnet")) / self.dataset
        self.work_root = Path(self.model_cfg.get("work_dir", "outputs/pgtnet_work")) / self.dataset
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        # Cache project root for resolving relative paths
        self._project_root = Path(__file__).resolve().parents[2]

    # -------------- Public API --------------
    def run_conversion(self) -> ExternalCallResult:
        """Run PGTNet converter once per dataset (idempotent on their side via --overwrite)."""
        pgtnet_repo = Path(self.model_cfg.get("pgtnet_repo", "third_party/PGTNet"))
        python = self._resolve_python_path(self.model_cfg.get("python", "python3"))
        # Verify interpreter exists/works; provide actionable error instead of crashing
        if not self._python_exists(python):
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    f"Configured Python interpreter not found or not runnable: {python}\n"
                    "Please provision the GraphGPS/PGTNet environment and point to it, e.g.:\n"
                    "  make graphgps_env\n"
                    "then rerun with +model.pgtnet.python=third_party/graphgps_venv/bin/python\n"
                    "Alternatively, set model.pgtnet.python to an absolute path of a Python with PyYAML, pm4py, scikit-learn, pandas installed."
                ),
                returncode=127,
                cmd=[str(python), "-V"],
            )
        converter = self.model_cfg.get("converter", {})
        cfg_dir = converter.get("config_dir", "conversion_configs")
        cfg_name = converter.get("config_name", "example.yaml")
        overwrite = str(converter.get("overwrite", False)).lower()
        input_xes = converter.get("input_xes")

        # Preflight checks
        script_path = pgtnet_repo / "GTconvertor.py"
        cfg_path = pgtnet_repo / cfg_dir / cfg_name
        if not script_path.exists():
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    f"PGTNet converter script not found at {script_path}.\n"
                    f"Configure model.pgtnet.pgtnet_repo to point to your PGTNet clone."
                ),
                returncode=2,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )
        if not cfg_path.exists():
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    f"PGTNet conversion config not found: {cfg_path}.\n"
                    f"Set model.pgtnet.converter.config_dir/name to a valid file relative to the PGTNet repo."
                ),
                returncode=2,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )
        # If the configured python lacks required modules but the helper env is available and healthy, auto-fallback to helper.
        try:
            helper = (self._project_root / "third_party/graphgps_venv/bin/python").resolve()
            if (not self._python_has_module(python, "yaml") or not self._python_has_module(python, "pm4py")) \
               and helper.exists() and os.access(helper, os.X_OK) \
               and self._python_has_module(str(helper), "yaml") and self._python_has_module(str(helper), "pm4py"):
                python = str(helper)
        except Exception:
            pass

        # Dependency preflight: PyYAML ('yaml') and pm4py are required by GTconvertor.py
        if not self._python_has_module(python, "yaml"):
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    "Missing dependency for conversion: 'PyYAML' (module 'yaml') is not importable in the configured "
                    f"Python interpreter ({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install pyyaml\n"
                    "Or run: make graphgps_env (which provisions a helper venv) and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python."
                ),
                returncode=3,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )
        if not self._python_has_module(python, "pm4py"):
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    "Missing dependency for conversion: 'pm4py' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install pm4py\n"
                    "Or run: make graphgps_env (which provisions a helper venv) and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python."
                ),
                returncode=3,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )
        if not self._python_has_module(python, "sklearn"):
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    "Missing dependency for conversion: 'scikit-learn' (module 'sklearn') is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install \"scikit-learn>=1.4,<1.6\"\n"
                    "Or run: make graphgps_env (which provisions a helper venv) and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python."
                ),
                returncode=3,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )
        if not self._python_has_module(python, "pandas"):
            return self._error_result(
                fold_idx=None,
                phase="convert",
                message=(
                    "Missing dependency for conversion: 'pandas' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install pandas==2.3.1\n"
                    "Or run: make graphgps_env (which provisions a helper venv) and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python."
                ),
                returncode=3,
                cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
            )

        # Prepare normalized conversion YAML to respect case attribute naming rules
        # - case_attributes and case_num_att in YAML must be specified WITHOUT the 'case:' prefix.
        #   If the provided YAML includes 'case:' accidentally, strip it before invoking GTconvertor.
        tmp_cfg_dir = self.work_root / "conversion_configs"
        tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_name_to_use = cfg_name
        cfg_dir_to_use = cfg_dir
        dataset_name_in_yaml = None
        try:
            yml_raw = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True) or {}
            # Normalize case attributes: strip any accidental 'case:' prefix
            def _strip_case_prefix_list(lst):
                out = []
                for it in (lst or []):
                    try:
                        s = str(it)
                        if s.startswith("case:"):
                            s = s[len("case:"):]
                        out.append(s)
                    except Exception:
                        out.append(it)
                return out
            changed = False
            yml_norm = dict(yml_raw)
            for k in ("case_attributes", "case_num_att"):
                if k in yml_norm and isinstance(yml_norm[k], (list, tuple)):
                    stripped = _strip_case_prefix_list(list(yml_norm[k]))
                    if stripped != yml_norm[k]:
                        yml_norm[k] = stripped
                        changed = True
            dataset_name_in_yaml = yml_norm.get("dataset_name")
            # Always write a temp YAML to ensure absolute path usage and immutability of vendor file
            tmp_cfg_path = tmp_cfg_dir / cfg_name
            with open(tmp_cfg_path, "w") as f:
                yaml.safe_dump(yml_norm, f, sort_keys=False)
            cfg_dir_to_use = str(tmp_cfg_dir)
            cfg_name_to_use = cfg_name
        except Exception:
            # Fallback: use original YAML as-is
            try:
                yml = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True) or {}
                dataset_name_in_yaml = yml.get("dataset_name")
            except Exception:
                dataset_name_in_yaml = None
            cfg_dir_to_use = cfg_dir
            cfg_name_to_use = cfg_name

        # Resolve/copy XES into PGTNet/raw_dataset/<dataset_name from YAML>
        xes_expected_path = None
        xes_columns_probe: Optional[List[str]] = None
        try:
            # Build candidate source paths
            candidates = []
            provided_src = None
            if input_xes:
                provided_src = Path(str(input_xes))
                if provided_src.is_dir():
                    # If a directory was passed, look for matching filenames inside
                    for ext in (".xes", ".xes.gz"):
                        if dataset_name_in_yaml:
                            candidates.append(provided_src / str(dataset_name_in_yaml))
                        candidates.append(provided_src / f"{self.dataset}{ext}")
                else:
                    candidates.append(provided_src)
            # Common fallback locations if nothing provided or provided one is missing
            raw_dir = pgtnet_repo / "raw_dataset"
            if dataset_name_in_yaml:
                candidates.append(raw_dir / str(dataset_name_in_yaml))
            for ext in (".xes", ".xes.gz"):
                candidates.append(self._project_root / "data" / "raw" / f"{self.dataset}{ext}")
                if dataset_name_in_yaml and not str(dataset_name_in_yaml).endswith(ext):
                    # dataset_name_in_yaml may already include extension; only add if different
                    candidates.append(self._project_root / "data" / "raw" / str(dataset_name_in_yaml))
            # Pick the first existing path
            src_path = next((p for p in candidates if p and Path(p).exists()), None)
            if src_path is None:
                # Construct a helpful error only if user explicitly set input_xes or there's no file in raw_dataset
                checked = "\n".join([str(p) for p in candidates])
                hint = (
                    f"Could not locate an input XES for conversion. Checked the following locations:\n{checked}\n"
                    f"Solutions:\n"
                    f"  - Place the XES file under {raw_dir} with the exact name from {cfg_dir}/{cfg_name} (dataset_name).\n"
                    f"  - Or set +model.pgtnet.converter.input_xes=/absolute/path/to/YourDataset.xes"
                )
                if provided_src is not None and not provided_src.exists():
                    return self._error_result(
                        fold_idx=None,
                        phase="convert",
                        message=(
                            f"Configured model.pgtnet.converter.input_xes does not exist: {provided_src}\n" + hint
                        ),
                        returncode=2,
                        cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
                    )
                else:
                    return self._error_result(
                        fold_idx=None,
                        phase="convert",
                        message=hint,
                        returncode=2,
                        cmd=[str(python), str(script_path), cfg_dir, cfg_name, "--overwrite", overwrite],
                    )
            # Copy/sync into PGTNet raw_dataset under the exact dataset name expected by PGTNet
            if dataset_name_in_yaml:
                dest_dir = raw_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / str(dataset_name_in_yaml)
                xes_expected_path = str(dest_path)
                # Force overwrite to avoid any stale/mtime edge cases
                try:
                    shutil.copyfile(src_path, dest_path)
                except Exception:
                    # If overwrite fails (e.g., permissions), fall back to conditional copy
                    try:
                        if (not dest_path.exists()) or (os.path.getmtime(dest_path) < os.path.getmtime(src_path)):
                            shutil.copyfile(src_path, dest_path)
                    except Exception:
                        pass
                # Probe available columns from the XES using the same interpreter, to help error messages later
                try:
                    probe_script = (
                        "import json,pm4py,sys; df=pm4py.read_xes(sys.argv[1]);"
                        "cols=list(df.columns); print(json.dumps({'columns':cols}))"
                    )
                    probe_cmd = [str(python), "-c", probe_script, str(dest_path)]
                    probe = subprocess.run(probe_cmd, cwd=str(pgtnet_repo), capture_output=True, text=True)
                    if probe.returncode == 0 and probe.stdout.strip():
                        import json as _json
                        payload = _json.loads(probe.stdout.strip())
                        xes_columns_probe = list(payload.get('columns') or [])
                        # Persist for user inspection
                        try:
                            (self.work_root / "shared").mkdir(parents=True, exist_ok=True)
                            with open(self.work_root / "shared" / "xes_columns.json", "w") as f:
                                _json.dump(payload, f, indent=2)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # Non-fatal; continue to try conversion
            pass

        cmd = [python, str(script_path), str(cfg_dir_to_use), str(cfg_name_to_use), "--overwrite", overwrite]
        res = self._exec(cmd, fold_idx=None, phase="convert", cwd=pgtnet_repo)
        # Detect silent FileNotFound from GTconvertor and surface as failure for clarity
        try:
            if res.stdout_path.exists():
                content = res.stdout_path.read_text(errors="ignore")
                if "File not found. Please provide a valid file path." in content:
                    # Retry once with the original YAML location in case relative path assumptions exist
                    try:
                        retry_cmd = [python, str(script_path), str(cfg_dir), str(cfg_name), "--overwrite", overwrite]
                        res_retry = self._exec(retry_cmd, fold_idx=None, phase="convert", cwd=pgtnet_repo)
                        if res_retry.returncode == 0 and res_retry.stdout_path.exists():
                            retry_out = res_retry.stdout_path.read_text(errors="ignore")
                            if "File not found. Please provide a valid file path." not in retry_out:
                                return res_retry
                    except Exception:
                        pass
                    # Append context to stderr
                    try:
                        with open(res.stderr_path, "a") as errf:
                            # Compute the path GTconvertor will try to open
                            expected_path = str((pgtnet_repo / 'raw_dataset' / (dataset_name_in_yaml or 'UNKNOWN')).resolve())
                            exists_flag = os.path.exists(expected_path)
                            errf.write(
                                "PGTNet GTconvertor reported missing input file.\n"
                                f"Expected an XES file named as specified in {cfg_dir_to_use}/{cfg_name_to_use} under {pgtnet_repo/'raw_dataset'}.\n"
                                f"Checked path: {expected_path} (exists={exists_flag})\n"
                                "Solutions:\n"
                                f"  - Place the XES file there, or\n  - Set model.pgtnet.converter.input_xes to an absolute path to your XES file.\n"
                            )
                    except Exception:
                        pass
                    # Return a failed result preserving logs (use rc=2 to signal input issue)
                    return ExternalCallResult(cmd=res.cmd, returncode=2, seconds=res.seconds, stdout_path=res.stdout_path, stderr_path=res.stderr_path)
        except Exception:
            pass
        # Provide clearer guidance when a KeyError suggests a missing attribute/column in XES
        try:
            if res.returncode != 0 and res.stderr_path.exists():
                serr = res.stderr_path.read_text(errors="ignore")
                import re
                m = re.search(r"KeyError: '([^']+)'", serr)
                if m:
                    missing = m.group(1)
                    with open(res.stderr_path, "a") as errf:
                        errf.write("\nPGTNet conversion failed due to a missing attribute/column: '" + missing + "'\n")
                        if xes_expected_path:
                            errf.write(f"XES inspected: {xes_expected_path}\n")
                        if xes_columns_probe:
                            # show a short subset and save full list in work dir
                            preview = ", ".join(xes_columns_probe[:20])
                            errf.write(f"Columns available in XES (first 20): {preview}\n")
                            if len(xes_columns_probe) > 20:
                                errf.write(f"(+{len(xes_columns_probe)-20} more; see {self.work_root/'shared'/'xes_columns.json'})\n")
                        errf.write(
                            "Check your conversion YAML's event_attributes/case_attributes.\n"
                            "- event_attributes must correspond exactly to event-level columns in the XES DataFrame.\n"
                            "- case_attributes in YAML must be listed WITHOUT 'case:'; the converter will prefix them internally.\n"
                        )
        except Exception:
            pass
        return res

    def run_train(self, fold_idx: int, seed: int) -> ExternalCallResult:
        graphgps_repo = Path(self.model_cfg.get("graphgps_repo", "third_party/GraphGPS"))
        pgtnet_repo = Path(self.model_cfg.get("pgtnet_repo", "third_party/PGTNet"))
        python = self._resolve_python_path(self.model_cfg.get("python", "python3"))
        if not self._python_exists(python):
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    f"Configured Python interpreter not found or not runnable: {python}\n"
                    "Please provision the environment (make graphgps_env) or set +model.pgtnet.python to a valid interpreter with torch, torch_geometric, yacs installed."
                ),
                returncode=127,
                cmd=[str(python), "-V"],
            )
        train_cfg = self.model_cfg.get("train", {})
        cfg_file = train_cfg.get("cfg_file")
        if not cfg_file:
            raise ValueError("model.pgtnet.train.cfg_file is required")
        main_path = graphgps_repo / "main.py"
        # Resolve cfg path: try GraphGPS repo, then PGTNet repo, then PGTNet/training_configs, then absolute
        cfg_candidates = []
        cfg_str = str(cfg_file)
        cfg_candidates.append(graphgps_repo / cfg_str)
        cfg_candidates.append(pgtnet_repo / cfg_str)
        cfg_candidates.append(pgtnet_repo / "training_configs" / cfg_str)
        abs_path = Path(cfg_str)
        if abs_path.is_absolute():
            cfg_candidates.insert(0, abs_path)
        cfg_path = next((p for p in cfg_candidates if p.exists()), None)
        if not main_path.exists():
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    f"GraphGPS main.py not found at {main_path}.\n"
                    f"Configure model.pgtnet.graphgps_repo to point to your GraphGPS clone."
                ),
                returncode=2,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_file), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        if cfg_path is None:
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    "Train cfg file not found. Tried the following locations:\n" +
                    "\n".join([str(p) for p in cfg_candidates]) +
                    "\nSet model.pgtnet.train.cfg_file to a valid path (can be absolute, relative to GraphGPS repo, or relative to PGTNet repo)."
                ),
                returncode=2,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_file), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        # Dependency preflight: torch_geometric and yacs must be importable in the configured Python
        if not self._python_has_module(python, "torch_geometric"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    "Missing dependency: 'torch_geometric' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install PyTorch Geometric and its extensions in that environment or run: "
                    "make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        if not self._python_has_module(python, "yacs"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    "Missing dependency: 'yacs' (used by torch_geometric.graphgym for configuration) is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install yacs\n"
                    "Or run: make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        # GraphGym also expects pytorch_lightning at import time
        if not self._python_has_module(python, "pytorch_lightning"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    "Missing dependency: 'pytorch_lightning' required by torch_geometric.graphgym is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install pytorch_lightning==2.5.3\n"
                    "Or run: make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        # torch_scatter is imported by GraphGPS encoders (e.g., SignNetNodeEncoder) and must be present
        if not self._python_has_module(python, "torch_scatter"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="train",
                message=(
                    "Missing dependency: 'torch_scatter' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install PyTorch Geometric extensions in that environment, e.g.:\n"
                    f"  {python} -m pip install -f https://data.pyg.org/whl/torch-2.8.0+cpu.html torch_scatter\n"
                    "Or run: make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        cmd = [
            python,
            str(main_path),
            "--cfg",
            str(cfg_path.resolve()),
            "run_multiple_splits",
            f"[{fold_idx}]",
            "seed",
            str(seed),
        ]
        return self._exec(cmd, fold_idx=fold_idx, phase="train", cwd=graphgps_repo)

    def run_infer(self, fold_idx: int, seed: int) -> ExternalCallResult:
        graphgps_repo = Path(self.model_cfg.get("graphgps_repo", "third_party/GraphGPS"))
        pgtnet_repo = Path(self.model_cfg.get("pgtnet_repo", "third_party/PGTNet"))
        python = self._resolve_python_path(self.model_cfg.get("python", "python3"))
        if not self._python_exists(python):
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    f"Configured Python interpreter not found or not runnable: {python}\n"
                    "Please provision the environment (make graphgps_env) or set +model.pgtnet.python to a valid interpreter with torch, torch_geometric, yacs installed."
                ),
                returncode=127,
                cmd=[str(python), "-V"],
            )
        infer_cfg = self.model_cfg.get("infer", {})
        cfg_file = infer_cfg.get("cfg_file")
        if not cfg_file:
            raise ValueError("model.pgtnet.infer.cfg_file is required")
        main_path = graphgps_repo / "main.py"
        # Resolve cfg path: try GraphGPS repo, then PGTNet repo, then PGTNet/inference_configs, then absolute
        cfg_candidates = []
        cfg_str = str(cfg_file)
        cfg_candidates.append(graphgps_repo / cfg_str)
        cfg_candidates.append(pgtnet_repo / cfg_str)
        cfg_candidates.append(pgtnet_repo / "inference_configs" / cfg_str)
        abs_path = Path(cfg_str)
        if abs_path.is_absolute():
            cfg_candidates.insert(0, abs_path)
        cfg_path = next((p for p in cfg_candidates if p.exists()), None)
        if not main_path.exists():
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    f"GraphGPS main.py not found at {main_path}.\n"
                    f"Configure model.pgtnet.graphgps_repo to point to your GraphGPS clone."
                ),
                returncode=2,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_file), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        if cfg_path is None:
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    "Infer cfg file not found. Tried the following locations:\n" +
                    "\n".join([str(p) for p in cfg_candidates]) +
                    "\nSet model.pgtnet.infer.cfg_file to a valid path (can be absolute, relative to GraphGPS repo, or relative to PGTNet repo)."
                ),
                returncode=2,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_file), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        # Dependency preflight: torch_geometric and yacs must be importable in the configured Python
        if not self._python_has_module(python, "torch_geometric"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    "Missing dependency: 'torch_geometric' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install PyTorch Geometric and its extensions in that environment or run: "
                    "make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        if not self._python_has_module(python, "yacs"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    "Missing dependency: 'yacs' (used by torch_geometric.graphgym for configuration) is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install it into that environment, e.g.:\n"
                    f"  {python} -m pip install yacs\n"
                    "Or run: make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        # Some GraphGPS encoders require torch_scatter (imported by signnet_pos_encoder)
        if not self._python_has_module(python, "torch_scatter"):
            return self._error_result(
                fold_idx=fold_idx,
                phase="infer",
                message=(
                    "Missing dependency: 'torch_scatter' is not importable in the configured Python interpreter "
                    f"({python}).\n"
                    "Install wheels compatible with your torch version, e.g.:\n"
                    f"  {python} -m pip install -f https://data.pyg.org/whl/torch-2.8.0+cpu.html torch_scatter\n"
                    "Or run: make graphgps_env and pass +model.pgtnet.python=third_party/graphgps_venv/bin/python"
                ),
                returncode=3,
                cmd=[str(python), str(main_path), "--cfg", str(cfg_str), "run_multiple_splits", f"[{fold_idx}]", "seed", str(seed)],
            )
        cmd = [
            python,
            str(main_path),
            "--cfg",
            str(cfg_path.resolve()),
            "run_multiple_splits",
            f"[{fold_idx}]",
            "seed",
            str(seed),
        ]
        # Provide pretrained.dir through environment variable if needed; most configs point to it internally
        return self._exec(cmd, fold_idx=fold_idx, phase="infer", cwd=graphgps_repo)

    def run_result_handler(self, dataset_key: Optional[str], seed: int, infer_cfg_stem: Optional[str]) -> Optional[ExternalCallResult]:
        """Optionally run PGTNet's ResultHandler.py to reconstruct identities and predictions mapping.
        Returns ExternalCallResult or None if not configured.
        """
        if not dataset_key or not infer_cfg_stem:
            return None
        pgtnet_repo = Path(self.model_cfg.get("pgtnet_repo", "third_party/PGTNet"))
        python = self._resolve_python_path(self.model_cfg.get("python", "python3"))
        if not self._python_exists(python):
            return self._error_result(
                fold_idx=None,
                phase="result_handler",
                message=(
                    f"Configured Python interpreter not found or not runnable: {python}\n"
                    "Please provision the environment (make graphgps_env) or set +model.pgtnet.python to a valid interpreter."
                ),
                returncode=127,
                cmd=[str(python), "-V"],
            )
        cfg_name = Path(infer_cfg_stem).stem
        cmd = [
            python,
            str(pgtnet_repo / "ResultHandler.py"),
            "--dataset_name",
            str(dataset_key),
            "--seed_number",
            str(seed),
            "--inference_config",
            str(cfg_name),
        ]
        return self._exec(cmd, fold_idx=None, phase="result_handler", cwd=pgtnet_repo)

    def run_full_fold(self, fold_idx: int, seed: int, execute: bool = True) -> Dict[str, Any]:
        """Orchestrate convert->train->infer for a single fold.
        Returns a manifest-like dict with an overall_success flag.
        Early-stops the pipeline if a prior phase fails (non-zero return code).
        """
        fold_dir = self.outputs_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": self.dataset,
            "fold": fold_idx,
            "seed": seed,
            "third_party_commits": self._read_third_party_commits(),
            "calls": [],
            "overall_success": True,
        }
        if not execute:
            self._write_json(fold_dir / "run_manifest.json", manifest)
            return manifest

        # Convert (one-time; allow repeated invocations)
        conv = self.run_conversion()
        manifest["calls"].append(self._call_to_dict("convert", conv))
        if int(conv.returncode) != 0:
            manifest["overall_success"] = False
            self._write_json(fold_dir / "run_manifest.json", manifest)
            return manifest

        # Train
        train = self.run_train(fold_idx, seed)
        manifest["calls"].append(self._call_to_dict("train", train))
        if int(train.returncode) != 0:
            manifest["overall_success"] = False
            self._write_json(fold_dir / "run_manifest.json", manifest)
            return manifest

        # Infer
        infer = self.run_infer(fold_idx, seed)
        manifest["calls"].append(self._call_to_dict("infer", infer))
        if int(infer.returncode) != 0:
            manifest["overall_success"] = False
            self._write_json(fold_dir / "run_manifest.json", manifest)
            return manifest

        # Optional: result handler
        dataset_key = self.model_cfg.get("dataset_key")
        infer_cfg = self.model_cfg.get("infer", {}).get("cfg_file")
        res = self.run_result_handler(dataset_key, seed, infer_cfg) if dataset_key and infer_cfg else None
        if res is not None:
            manifest["calls"].append(self._call_to_dict("result_handler", res))
            if int(res.returncode) != 0:
                manifest["overall_success"] = False

        # Save manifest
        self._write_json(fold_dir / "run_manifest.json", manifest)
        return manifest

    # -------------- Internals --------------
    def _resolve_python_path(self, python: str) -> str:
        """Resolve configured python to an absolute path with sensible defaults.
        Preference order:
        1) If the path looks like uv's managed interpreter and the helper env exists → use helper env.
        2) Absolute path provided → return as-is.
        3) Relative path containing separators → resolve relative to project root if exists.
        4) Auto-detect helper env at third_party/graphgps_venv/bin/python if config left as default 'python'/'python3'.
        5) Fallback to provided value (e.g., 'python3').
        """
        try:
            helper = (self._project_root / "third_party/graphgps_venv/bin/python").resolve()
            p_str = str(python)
            # Heuristic: if uv's shim is selected, prefer the dedicated helper env to avoid missing pm4py/yaml
            if "/.local/share/uv/python/" in p_str and helper.exists() and os.access(helper, os.X_OK):
                return str(helper)
            p = Path(p_str)
            if p.is_absolute():
                return str(p)
            # Heuristic: treat values containing os separators as path-like
            if any(sep in p_str for sep in (os.sep, '/')):
                candidate = (self._project_root / p).resolve()
                if candidate.exists():
                    return str(candidate)
            # Auto-detect helper env when user did not override and default interpreter would be used
            norm = p_str.strip().lower()
            if norm in {"python", "python3", "python3.11"} and helper.exists() and os.access(helper, os.X_OK):
                return str(helper)
        except Exception:
            pass
        return str(python)

    def _exec(self, cmd: List[str], fold_idx: Optional[int], phase: str, cwd: Optional[Path] = None) -> ExternalCallResult:
        fold_dir = self.outputs_root / (f"fold_{fold_idx}" if fold_idx is not None else "shared")
        fold_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = fold_dir / f"{phase}.stdout.log"
        stderr_path = fold_dir / f"{phase}.stderr.log"
        timeout = int(self.model_cfg.get("subprocess_timeout", 0) or 0)
        start = time.time()
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            # Use text-mode capture
            try:
                proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=None if timeout <= 0 else timeout, cwd=str(cwd) if cwd else None)
                rc = int(proc.returncode)
            except subprocess.TimeoutExpired:
                rc = 124
                err.write("Subprocess timed out. You can adjust model.pgtnet.subprocess_timeout (0 means no timeout).\n")
            except FileNotFoundError as e:
                rc = 127
                err.write(
                    (
                        f"Failed to execute external command. File not found: {e.filename}\n"
                        f"Command: {' '.join(shlex.quote(str(c)) for c in cmd)}\n"
                        f"Working dir: {cwd if cwd else os.getcwd()}\n"
                        "If this refers to the Python interpreter, provision it with `make graphgps_env` and rerun with\n"
                        "+model.pgtnet.python=third_party/graphgps_venv/bin/python, or set model.pgtnet.python to a valid Python path.\n"
                    )
                )
        seconds = time.time() - start
        return ExternalCallResult(cmd=cmd, returncode=rc, seconds=seconds, stdout_path=stdout_path, stderr_path=stderr_path)

    def _error_result(self, fold_idx: Optional[int], phase: str, message: str, returncode: int = 2, cmd: Optional[List[str]] = None) -> ExternalCallResult:
        """Synthesize a failed ExternalCallResult without executing a subprocess.
        Writes the error message to the phase stderr log for easier debugging.
        """
        fold_dir = self.outputs_root / (f"fold_{fold_idx}" if fold_idx is not None else "shared")
        fold_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = fold_dir / f"{phase}.stdout.log"
        stderr_path = fold_dir / f"{phase}.stderr.log"
        try:
            with open(stderr_path, "w") as err:
                err.write(message.strip() + "\n")
        except Exception:
            pass
        return ExternalCallResult(cmd=cmd or [], returncode=int(returncode), seconds=0.0, stdout_path=stdout_path, stderr_path=stderr_path)

    def _python_exists(self, python: str) -> bool:
        """Return True if the given python interpreter path is runnable.
        We attempt to execute `<python> -V`. Returns False on FileNotFoundError or non-zero exit.
        """
        try:
            res = subprocess.run([python, "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _python_has_module(self, python: str, module: str) -> bool:
        """Return True if the given python interpreter can import the module.
        Avoids importlib.util to guard against third-party 'importlib' shadowing.
        """
        try:
            # Use __import__ to probe availability; exit code 0 if import succeeds, else 3
            code = (
                "import sys; m='" + module + "';\n"
                "try:\n"
                "    __import__(m)\n"
                "    sys.exit(0)\n"
                "except Exception:\n"
                "    sys.exit(3)\n"
            )
            res = subprocess.run(
                [python, "-c", code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _read_third_party_commits(self) -> Dict[str, Any]:
        commits = {}
        try:
            g = Path("third_party/GraphGPS.COMMIT")
            if g.exists():
                commits["GraphGPS"] = g.read_text().strip()
        except Exception:
            pass
        try:
            p = Path("third_party/PGTNet.COMMIT")
            if p.exists():
                commits["PGTNet"] = p.read_text().strip()
        except Exception:
            pass
        return commits

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _call_to_dict(self, name: str, res: ExternalCallResult) -> Dict[str, Any]:
        return {
            "name": name,
            "cmd": res.cmd,
            "returncode": res.returncode,
            "seconds": res.seconds,
            "stdout": str(res.stdout_path),
            "stderr": str(res.stderr_path),
        }
