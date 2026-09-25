#!/usr/bin/env python3
"""Create and verify patient-isolated ADNI manifests without modifying images.

Requires Python >= 3.9 and Pillow. See DATA_PROTOCOL.md for training safeguards
that these checks cannot enforce. A patient may have several scans, and each
scan contains several JPEG slices; the patient is the unit of every split.
Checks cover supplied identifiers and exact image duplicates, not approximate
duplicates, upstream preprocessing, or future model-training behavior.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile

VERSION = 1
SOURCE_FIELDS = [
    "relative_path", "original_split", "folder_label", "label", "metadata_label",
    "patient_id", "image_id", "slice_index", "width", "height", "mode",
    "file_bytes", "file_sha256", "pixel_sha256",
]
FIELDS = SOURCE_FIELDS + ["patient_stratum", "partition", "fold"]
PATIENT_FIELDS = ["patient_id", "stratum", "labels", "num_scans", "num_images", "partition", "fold"]
# Each pair is (original metadata label, binary training label): AD 2 -> 1,
# NC 0 -> 0. Keep both values in the manifests so the mapping remains auditable.
SOURCE_LABELS = {"AD": (2, 1), "NC": (0, 0)}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d+)_")
IMAGE_PATTERN = re.compile(r"_I(\d+)\.nii(?:\.gz)?$")


class AuditError(ValueError):
    """Input data or manifests violate the declared protocol."""


def require(condition, message):
    """Stop the audit when an input or split violates a required condition."""
    if not condition:
        raise AuditError(message)


def digest(data):
    """Return a SHA-256 fingerprint for an exact sequence of bytes."""
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    """Serialize a value consistently for reproducible fingerprints."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_json(path):
    """Read JSON, accepting an optional UTF-8 byte-order mark."""
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    """Write an indented, UTF-8 audit record."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def profiles_for(rows):
    """Summarize each patient's scan labels without changing any scan's label.

    A longitudinal patient can have both NC and AD scans. Such patients use
    the mixed stratum and still stay together during every partition.
    """
    profiles = {}
    for row in rows:
        profile = profiles.setdefault(row["patient_id"], {"labels": set(), "scans": set(), "images": 0})
        profile["labels"].add(int(row["label"]))
        profile["scans"].add(row["image_id"])
        profile["images"] += 1
    for profile in profiles.values():
        profile["stratum"] = {frozenset({0}): "NC_only", frozenset({1}): "AD_only",
                              frozenset({0, 1}): "mixed"}[frozenset(profile["labels"])]
    return profiles


def inventory(data_root, expected_slices):
    """Audit source identities, labels, scan completeness, and exact duplicates.

    Original train/test folders provide provenance only; their patient overlap
    is measured here and corrected by the new patient-based assignments.
    """
    try:
        import PIL
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise AuditError("缺少 Pillow。请在当前 Python 环境安装 Pillow 后运行。") from exc

    metadata_path = data_root / "meta_data_with_label.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
    require(isinstance(metadata, dict), "元数据顶层必须是字典。")
    rows, errors, ignored = [], [], []
    seen_slices = set()
    identities = {}
    pixel_owners = {}
    within_patient_duplicates = 0

    for split in ("train", "test"):
        for folder_label, (metadata_label, label) in SOURCE_LABELS.items():
            folder = data_root / "AD_NC" / split / folder_label
            require(folder.is_dir(), f"缺少源目录: {folder}")
            files = sorted(path for path in folder.rglob("*") if path.is_file())
            jpeg_files = [path for path in files if path.suffix.lower() in (".jpeg", ".jpg")]
            ignored.extend(str(path.relative_to(data_root)) for path in files
                           if path.suffix.lower() not in (".jpeg", ".jpg"))
            require(jpeg_files, f"目录没有 JPEG 图片: {folder}")
            for path in jpeg_files:
                relative = path.relative_to(data_root).as_posix()
                try:
                    match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
                    require(match is not None, f"文件名无法解析: {relative}")
                    image_id, slice_text = match.groups()
                    slice_index = int(slice_text)
                    record = metadata.get(image_id)
                    require(isinstance(record, dict), f"JSON 缺少有效影像记录 {image_id}: {relative}")
                    require(type(record.get("label")) is int and record["label"] == metadata_label,
                            f"目录/JSON 标签冲突: {relative}, JSON label={record.get('label')!r}")
                    raw = record.get("raw")
                    require(isinstance(raw, str), f"影像 {image_id} 缺少 raw 路径")
                    patient_match = PATIENT_PATTERN.search(raw)
                    source_match = IMAGE_PATTERN.search(raw)
                    require(patient_match is not None and source_match is not None,
                            f"元数据路径编号无法解析: {raw}")
                    require(source_match.group(1) == image_id, f"JSON 键和路径影像编号不一致: {image_id}")
                    patient_id = patient_match.group(1)
                    # Available derivative paths must refer to the same patient
                    # and scan as raw; missing optional paths are not required.
                    for field in ("c1", "c2", "c3", "c4", "c5", "masked"):
                        value = record.get(field)
                        if value is None:
                            continue
                        require(isinstance(value, str), f"{image_id}: {field} 不是路径字符串")
                        pm, im = PATIENT_PATTERN.search(value), IMAGE_PATTERN.search(value)
                        require(pm is not None and im is not None and pm.group(1) == patient_id
                                and im.group(1) == image_id, f"{image_id}: {field} 与 raw 身份不一致")
                    identity = (patient_id, label)
                    require(image_id not in identities or identities[image_id] == identity,
                            f"同一影像映射到不同患者或标签: {image_id}")
                    identities[image_id] = identity
                    slice_key = (image_id, slice_index)
                    # Different filenames must not disguise the same scan/slice.
                    require(slice_key not in seen_slices, f"重复影像/切片编号: {slice_key}")
                    seen_slices.add(slice_key)

                    content = path.read_bytes()
                    with Image.open(io.BytesIO(content)) as original:
                        require(original.format == "JPEG", f"扩展名与实际格式不符: {relative}")
                        original.load()  # Detect truncated/undecodable JPEGs, not just header errors.
                        mode = original.mode
                        # Compare decoded pixels as well as file bytes: JPEG
                        # encodings or metadata can differ for identical pixels.
                        # This normalization supports hashing, not preprocessing
                        # for model input, and does not detect near-duplicates.
                        canonical = ImageOps.exif_transpose(original).convert("RGB")
                        width, height = canonical.size
                        pixel_hash = digest(json_bytes([width, height, "RGB"]) + b"\0" + canonical.tobytes())
                    previous = pixel_owners.get(pixel_hash)
                    if previous is not None:
                        previous_patient, previous_label, previous_path = previous
                        require(previous_label == label,
                                f"相同像素对应不同标签: {previous_path} <-> {relative}")
                        require(previous_patient == patient_id,
                                f"跨患者重复像素，需人工核查后再划分: {previous_path} <-> {relative}")
                        # Same-patient, same-label copies are recorded; keeping
                        # that patient together prevents them crossing roles.
                        within_patient_duplicates += 1
                    else:
                        pixel_owners[pixel_hash] = (patient_id, label, relative)
                    row = dict(zip(SOURCE_FIELDS, [relative, split, folder_label, label, metadata_label,
                               patient_id, image_id, slice_index, width, height, mode, len(content),
                               digest(content), pixel_hash]))
                    rows.append({key: str(value) for key, value in row.items()})
                except (AuditError, OSError, ValueError) as exc:
                    errors.append(f"{relative}: {exc}")
    if errors:
        raise AuditError(f"数据审计失败，共 {len(errors)} 项异常；前 12 项:\n" + "\n".join(errors[:12]))

    scans = Counter(row["image_id"] for row in rows)
    # image_id identifies a scan, whereas each row is one JPEG slice. The
    # expected count checks completeness; it does not infer slice anatomy.
    incomplete = {key: count for key, count in scans.items() if count != expected_slices}
    require(not incomplete, f"影像切片数应为 {expected_slices}，异常示例: {list(incomplete.items())[:10]}")
    rows.sort(key=lambda row: row["relative_path"])
    profiles = profiles_for(rows)
    old_patients = {split: {r["patient_id"] for r in rows if r["original_split"] == split}
                    for split in ("train", "test")}
    old_scans = {split: {r["image_id"] for r in rows if r["original_split"] == split}
                 for split in ("train", "test")}
    summary = {
        "images": len(rows), "image_records": len(scans), "patients": len(profiles),
        "patient_strata": dict(Counter(p["stratum"] for p in profiles.values())),
        "mixed_label_patients": sorted(p for p, profile in profiles.items() if profile["stratum"] == "mixed"),
        "original_patient_overlap": len(old_patients["train"] & old_patients["test"]),
        "original_image_overlap": len(old_scans["train"] & old_scans["test"]),
        "within_patient_same_label_duplicate_pixels": within_patient_duplicates,
        "ignored_non_jpeg_files": ignored,
        "metadata_sha256": digest(metadata_bytes), "source_fingerprint": digest(json_bytes(rows)),
        "python_version": sys.version.split()[0], "pillow_version": PIL.__version__,
        "pixel_hash_rule": "SHA256(dimensions + RGB decoded pixels after EXIF transpose); exact equality only",
    }
    return rows, summary


def stratified_take(patient_ids, count, profiles, rng):
    """Take a fixed patient count with largest-remainder stratum allocation.

    Strata are patient label histories, not slice counts. This is deliberately
    independent of scikit-learn's sample-weighted StratifiedGroupKFold.
    """
    patient_ids = sorted(patient_ids)
    require(0 < count < len(patient_ids), "患者子集太小，无法建立所要求的独立留出组。")
    buckets = defaultdict(list)
    for patient_id in patient_ids:
        buckets[profiles[patient_id]["stratum"]].append(patient_id)
    exact = {key: len(value) * count / len(patient_ids) for key, value in buckets.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    for key in sorted(buckets, key=lambda key: (-(exact[key] - quotas[key]), key))[:remainder]:
        quotas[key] += 1
    chosen = set()
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        chosen.update(buckets[key][:quotas[key]])
    return chosen, set(patient_ids) - chosen


def make_plan(rows, config):
    """Reserve test/calibration patients, then assign development CV roles.

    Patient-history strata balance patient counts, not slice counts. This is
    a custom deterministic splitter, not sklearn's StratifiedGroupKFold.
    """
    profiles = profiles_for(rows)
    rng = random.Random(config["seed"])
    patients = set(profiles)
    test_count = math.floor(len(patients) * config["test_fraction"] + 0.5)
    calibration_count = math.floor(len(patients) * config["calibration_fraction"] + 0.5)
    # Both held-out counts are fractions of all patients. Neither group enters
    # CV: calibration is reserved for thresholds after model selection, and
    # test is reserved for the final frozen pipeline's evaluation.
    test, remaining = stratified_take(patients, test_count, profiles, rng)
    calibration, development = stratified_take(remaining, calibration_count, profiles, rng)
    folds = config["folds"]
    require(len(development) >= folds, "开发集患者数量少于折数。")
    assignment = {patient: ("test", "") for patient in test}
    assignment.update({patient: ("calibration", "") for patient in calibration})
    loads = [0] * folds
    for stratum in sorted({p["stratum"] for p in profiles.values()}):
        bucket = sorted(p for p in development if profiles[p]["stratum"] == stratum)
        rng.shuffle(bucket)
        stratum_loads = [0] * folds
        for patient in bucket:
            best = min((stratum_loads[i], loads[i]) for i in range(folds))
            choices = [i for i in range(folds) if (stratum_loads[i], loads[i]) == best]
            fold = rng.choice(choices)
            assignment[patient] = ("development", str(fold + 1))
            stratum_loads[fold] += 1
            loads[fold] += 1
    early_stops = {}
    for fold in range(1, folds + 1):
        # The outer validation patients never select a stopping epoch. Draw a
        # separate early-stop group from this fold's training side instead.
        training_pool = {p for p in development if assignment[p][1] != str(fold)}
        count = math.floor(len(training_pool) * config["early_stop_fraction"] + 0.5)
        early, _ = stratified_take(training_pool, count, profiles, random.Random(config["seed"] + 1000 + fold))
        early_stops[str(fold)] = sorted(early)
    return assignment, early_stops


def patient_rows(rows):
    """Build a patient-level summary of assigned roles and scan/image counts."""
    profiles = profiles_for(rows)
    assignment = {row["patient_id"]: (row["partition"], row["fold"]) for row in rows}
    return [dict(zip(PATIENT_FIELDS, [patient, profiles[patient]["stratum"],
                 "|".join(map(str, sorted(profiles[patient]["labels"]))),
                 str(len(profiles[patient]["scans"])), str(profiles[patient]["images"]),
                 *assignment[patient]])) for patient in sorted(profiles)]


def partition_summary(rows):
    """Report patient, scan, slice, and label counts for one manifest."""
    profiles = profiles_for(rows)
    scans = {r["image_id"]: r for r in rows}
    return {"images": len(rows), "image_records": len(scans), "patients": len(profiles),
            "images_by_label": dict(Counter(r["folder_label"] for r in rows)),
            "image_records_by_label": dict(Counter(r["folder_label"] for r in scans.values())),
            "patient_strata": dict(Counter(p["stratum"] for p in profiles.values()))}


def artifacts_for(rows, config, early_stops):
    """Expand patient assignments into the complete set of CSV manifests.

    A development patient's fold identifies their one outer-validation turn.
    Their role can change to train or early_stop in another independently
    trained fold; isolation is required within each fold, not across all folds.
    """
    artifacts = {"all.csv": rows, "patients.csv": patient_rows(rows)}
    for partition in ("development", "calibration", "test"):
        artifacts[f"{partition}.csv"] = [r for r in rows if r["partition"] == partition]
    development = artifacts["development.csv"]
    for fold in range(1, config["folds"] + 1):
        early = set(early_stops[str(fold)])
        pool = [r for r in development if r["fold"] != str(fold)]
        prefix = f"fold_{fold:02d}"
        artifacts[f"{prefix}/train.csv"] = [r for r in pool if r["patient_id"] not in early]
        artifacts[f"{prefix}/early_stop.csv"] = [r for r in pool if r["patient_id"] in early]
        artifacts[f"{prefix}/val.csv"] = [r for r in development if r["fold"] == str(fold)]
    return artifacts


def check_boundaries(artifacts, config):
    """Check within-fold isolation, held-out isolation, and full CV coverage.

    These checks use manifest identifiers and exact content fingerprints.
    They cannot establish that upstream patient identifiers are correct.
    """
    def check(roles):
        """Require both classes and pairwise disjoint identities/content."""
        for name in roles:
            rows = artifacts[name]
            require(rows, f"划分为空: {name}")
            require({r["label"] for r in rows} == {"0", "1"},
                    f"{name} 未包含两种诊断，不能可靠计算所需分类指标。请检查患者规模/预先确定的比例。")
        for index, first in enumerate(roles):
            for second in roles[index + 1:]:
                for field in ("patient_id", "image_id", "relative_path", "file_sha256", "pixel_sha256"):
                    overlap = {r[field] for r in artifacts[first]} & {r[field] for r in artifacts[second]}
                    require(not overlap, f"边界重叠: {first} <-> {second}, {field}: {list(overlap)[:3]}")

    check(["development.csv", "calibration.csv", "test.csv"])
    all_val_paths = []
    development_paths = {r["relative_path"] for r in artifacts["development.csv"]}
    for fold in range(1, config["folds"] + 1):
        prefix = f"fold_{fold:02d}"
        roles = [f"{prefix}/{name}.csv" for name in ("train", "early_stop", "val")]
        check(roles + ["calibration.csv", "test.csv"])
        union = {r["relative_path"] for role in roles for r in artifacts[role]}
        require(union == development_paths, f"第 {fold} 折没有准确覆盖完整开发集。")
        all_val_paths.extend(r["relative_path"] for r in artifacts[f"{prefix}/val.csv"])
    require(set(all_val_paths) == development_paths and len(all_val_paths) == len(development_paths),
            "每张开发集图片必须且只能作为一次外层验证图片。")


def write_csv(path, rows, fields):
    """Write a manifest with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, fields):
    """Read a manifest only when its columns match the declared schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, f"CSV 字段不符合协议: {path}")
        return list(reader)


def config_from_args(args):
    """Validate the reproducible split settings used by prepare and verify."""
    config = {key: getattr(args, key) for key in ("seed", "folds", "test_fraction",
              "calibration_fraction", "early_stop_fraction", "expected_slices")}
    require(config["folds"] >= 2, "交叉验证至少需要 2 折。")
    require(config["expected_slices"] > 0, "expected-slices 必须大于零。")
    require(0 < config["test_fraction"] < 1 and 0 < config["calibration_fraction"] < 1
            and config["test_fraction"] + config["calibration_fraction"] < 1,
            "测试/校准比例必须大于零，且其和小于 1。")
    require(0 < config["early_stop_fraction"] < 0.5, "early-stop-fraction 应在 0 和 0.5 之间。")
    return config


def prepare(args):
    """Audit sources, create a new split, and publish only complete outputs."""
    root, output = args.data_root.resolve(), args.output.resolve()
    require(output != root and root not in output.parents, "输出目录必须在源数据目录之外。")
    require(not output.exists(), f"输出已存在，不会覆盖既定划分: {output}。请使用 verify 检查。")
    config = config_from_args(args)
    print("正在核对元数据并读取全部 JPEG 像素……", flush=True)
    source_rows, audit = inventory(root, config["expected_slices"])
    profiles = profiles_for(source_rows)
    assignment, early_stops = make_plan(source_rows, config)
    rows = [dict(row, patient_stratum=profiles[row["patient_id"]]["stratum"],
                 partition=assignment[row["patient_id"]][0], fold=assignment[row["patient_id"]][1])
            for row in source_rows]
    artifacts = artifacts_for(rows, config, early_stops)
    check_boundaries(artifacts, config)
    report = {
        "protocol_version": VERSION, "config": config, "data_root_at_creation": str(root),
        "source_audit": audit, "early_stop_patients": early_stops,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "stratification": "Patient histories: AD_only / NC_only / mixed; per-scan labels unchanged.",
        "partitions": {name: partition_summary(value) for name, value in artifacts.items()
                       if name not in ("all.csv", "patients.csv")},
        "checks": {"patient_image_path_and_exact_hash_boundaries": "passed",
                   "outer_validation_exactly_once_per_development_image": "passed"},
        "limitations": ["Checks use the patient identifiers supplied by the dataset.",
                        "Exact decoded duplicates are checked; near-duplicates are not exhaustively detected.",
                        "Upstream preprocessing provenance and future training code require separate review.",
                        "Fold models must be initialized independently; CSVs alone cannot enforce this."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary sibling directory first. Failed checks or writes
    # must not leave a partially populated directory that appears ready to use.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        for name, values in artifacts.items():
            write_csv(stage / name, values, PATIENT_FIELDS if name == "patients.csv" else FIELDS)
        write_json(stage / "report.json", report)
        checksums = {p.relative_to(stage).as_posix(): digest(p.read_bytes())
                     for p in sorted(stage.rglob("*")) if p.is_file()}
        write_json(stage / "COMPLETED.json", {"protocol_version": VERSION, "sha256": checksums})
        require(not output.exists(), f"输出目录在运行期间被创建，拒绝覆盖: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"原始 train/test 重叠患者: {audit['original_patient_overlap']}")
    print(f"审计后数据: {audit['patients']} 名患者 / {audit['image_records']} 个影像 / {audit['images']} 张图片")
    for name in ("development.csv", "calibration.csv", "test.csv"):
        value = report["partitions"][name]
        print(f"{name}: {value['patients']} 名患者, {value['images']} 张图片")
    print(f"已建立 {config['folds']} 折，所有患者/影像/精确重复边界检查通过。")
    print(f"输出: {output}\n请在训练前运行 verify；训练必须读取生成的清单。")


def verify(args):
    """Independently reread sources and reconstruct the recorded split.

    Checksums reveal changed outputs, but are not proof of correct identities.
    Therefore, do not trust patient IDs or assignments merely because they
    appear in a CSV: rebuild them from source metadata and decoded images.
    """
    root, output = args.data_root.resolve(), args.output.resolve()
    seal = read_json(output / "COMPLETED.json")
    require(seal.get("protocol_version") == VERSION, "不支持的清单协议版本。")
    checksums = seal.get("sha256")
    require(isinstance(checksums, dict), "完成标记缺少文件校验信息。")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p != output / "COMPLETED.json"}
    require(actual_files == set(checksums), "输出目录文件集合发生变化或不完整。")
    for relative, expected in checksums.items():
        path = output / relative
        require(path.resolve().is_relative_to(output), "完成标记含目录外路径。")
        require(digest(path.read_bytes()) == expected, f"输出文件被修改: {relative}")
    report = read_json(output / "report.json")
    require(report.get("protocol_version") == VERSION, "报告协议版本不一致。")
    config = config_from_args(argparse.Namespace(**report["config"]))
    print("正在重新读取真实数据、元数据和图片像素，独立核查清单……", flush=True)
    source_rows, audit = inventory(root, config["expected_slices"])
    for field in ("source_fingerprint", "metadata_sha256"):
        require(audit[field] == report["source_audit"][field], f"源数据/元数据已变化: {field}")
    assignment, early_stops = make_plan(source_rows, config)
    # Reproduce the frozen seed/protocol, then compare every manifest row.
    # Editing a patient ID and updating its checksum cannot bypass this check.
    require(early_stops == report["early_stop_patients"], "早停患者清单与固定随机种子不一致。")
    profiles = profiles_for(source_rows)
    expected_rows = [dict(row, patient_stratum=profiles[row["patient_id"]]["stratum"],
                         partition=assignment[row["patient_id"]][0], fold=assignment[row["patient_id"]][1])
                     for row in source_rows]
    expected_artifacts = artifacts_for(expected_rows, config, early_stops)
    require(set(checksums) == set(expected_artifacts) | {"report.json"}, "清单文件集合与协议不一致。")
    actual_artifacts = {}
    for name, expected in expected_artifacts.items():
        actual = read_csv(output / name, PATIENT_FIELDS if name == "patients.csv" else FIELDS)
        require(actual == expected, f"清单与真实身份、固定划分或完整覆盖不一致: {name}")
        actual_artifacts[name] = actual
    check_boundaries(actual_artifacts, config)
    for name, expected in expected_artifacts.items():
        if name not in ("all.csv", "patients.csv"):
            require(report["partitions"][name] == partition_summary(expected), f"统计报告与清单不一致: {name}")
    print("PASS：真实数据一致；患者/影像/路径/精确重复无跨组重叠；每折完整且每张开发图片恰好验证一次。")
    print("此结果不替代训练流程、近似重复或上游预处理审查。")


def main(argv=None):
    """Expose prepare/verify commands and return a nonzero status on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--seed", type=int, default=3710)
            child.add_argument("--folds", type=int, default=5)
            child.add_argument("--test-fraction", type=float, default=0.20)
            child.add_argument("--calibration-fraction", type=float, default=0.10)
            child.add_argument("--early-stop-fraction", type=float, default=0.10)
            child.add_argument("--expected-slices", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        {"prepare": prepare, "verify": verify}[args.command](args)
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
