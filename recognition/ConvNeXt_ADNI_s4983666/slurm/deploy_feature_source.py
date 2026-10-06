"""Verify and install a code-only bundle while preserving manifests, data and runs."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time


ROOT_FILES = {'README.md', '.gitignore', 'adni_splits.py', 'train.py', 'predict.py',
              'modules.py', 'dataset.py', 'diagnose.py', 'diagnose_followup.py',
              'run_experiment_suite.py', 'run_feature_experiments.py', 'train_feature_experiment.py',
              'recover_feature_report.py', 'audit_preprocessing.py'}
PACKAGES = {'models', 'dataset', 'engine', 'evaluation', 'utils', 'tests', 'slurm', 'docs', 'config'}
EXTENSIONS = {'.py', '.md', '.txt', '.json', '.sh', '.sbatch'}


def permitted(name: str) -> bool:
    """Allow only known code paths, never data, credentials, weights or run logs."""
    path = PurePosixPath(name)
    if name != path.as_posix() or path.is_absolute() or any(p in ('..', '.', '__pycache__') for p in path.parts):
        return False
    if name in ROOT_FILES:
        return True
    return len(path.parts) >= 2 and path.parts[0] in PACKAGES and path.suffix in EXTENSIONS


def checksum(path: Path) -> str:
    """Read a file fingerprint without loading patient data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_seals(splits: Path) -> dict:
    """Check every sealed manifest byte and the exact preregistered marker."""
    expected = 'd370391aacf381b9eabdff4b798f962b867fa5109f0b0d64f36ff0b8e663c206'
    marker = splits / 'COMPLETED.json'
    if checksum(marker) != expected:
        raise ValueError('Unexpected frozen manifest identity.')
    records = json.loads(marker.read_text())['sha256']
    for name, digest in records.items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Invalid sealed path.')
        if checksum(splits / name) != digest:
            raise ValueError(f'Frozen manifest failed its seal: {name}')
    return records | {'COMPLETED.json': expected}


def verify_bundle(archive: Path) -> tuple[dict, dict[str, bytes]]:
    """Validate member types, a closed file inventory and individual SHA-256s."""
    payload: dict[str, bytes] = {}
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in payload:
                raise ValueError('Bundle must contain unique regular files only.')
            if member.name != 'BUNDLE_MANIFEST.json' and not permitted(member.name):
                raise ValueError(f'Forbidden bundle member: {member.name}')
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError('Unreadable bundle member.')
            payload[member.name] = stream.read()
    manifest = json.loads(payload.pop('BUNDLE_MANIFEST.json').decode())
    if set(manifest['sha256']) != set(payload):
        raise ValueError('Bundle inventory differs from the manifest.')
    for name, content in payload.items():
        if hashlib.sha256(content).hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'Bundle checksum failed: {name}')
    return manifest, payload


def install(archive: Path, target: Path, splits: Path, *, check_jobs: bool = True) -> dict:
    """Back up replaced source and atomically replace files, with guarded rollback."""
    target, splits = target.absolute(), splits.resolve()
    if target.is_symlink() or not target.is_dir():
        raise ValueError('Target must be an existing nonsymlink source directory.')
    target = target.resolve()
    if target == splits or target in splits.parents or splits in target.parents:
        raise ValueError('Source and frozen split paths must be separate.')
    manifest, payload = verify_bundle(archive)
    before_seals = frozen_seals(splits)
    if check_jobs:
        jobs = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser()],
                              capture_output=True, text=True, check=True)
        if jobs.stdout.strip():
            raise ValueError('Active or pending jobs exist; keep source immutable while jobs run.')
    existing: dict[str, str | None] = {}
    for name in payload:
        path = target / name
        # No symlink can redirect a source write into protected storage.
        if any(p.is_symlink() for p in (path, *path.parents) if p == target or target in p.parents):
            raise ValueError(f'Symlink in source destination: {name}')
        if path.exists() and not path.is_file():
            raise ValueError(f'Existing source destination is not a file: {name}')
        existing[name] = checksum(path) if path.exists() else None
    suffix = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + checksum(archive)[:10]
    backup = target.parent / ('code_before_feature_v2_' + suffix + '.tar.gz')
    if backup.exists():
        raise ValueError('Backup name already exists; refusing overwrite.')
    with tarfile.open(backup, 'x:gz') as saved:
        for name, digest in existing.items():
            if digest is not None:
                saved.add(target / name, arcname=name, recursive=False)
    installed = []
    try:
        for name, content in payload.items():
            path = target / name
            if (checksum(path) if path.exists() else None) != existing[name]:
                raise ValueError(f'Source changed during installation: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.source_install_', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            installed.append(name)
        if any(checksum(target / n) != h for n, h in manifest['sha256'].items()):
            raise ValueError('Installed source verification failed.')
        if frozen_seals(splits) != before_seals:
            raise ValueError('Frozen manifest changed during installation.')
    except Exception:
        with tarfile.open(backup, 'r:gz') as saved:
            for name in installed:
                path = target / name
                if checksum(path) != manifest['sha256'][name]:
                    # Do not overwrite another process's intervening edit.
                    continue
                if existing[name] is None:
                    path.unlink()
                else:
                    stream = saved.extractfile(name)
                    if stream is None:
                        raise RuntimeError(f'Backup is incomplete: {name}')
                    path.write_bytes(stream.read())
        raise
    return {'status': 'source_installed_verified', 'target': str(target), 'backup': str(backup),
            'source_files': len(payload), 'archive_sha256': checksum(archive),
            'frozen_manifest_sha256': before_seals['COMPLETED.json'],
            'sealed_files_verified': len(before_seals), 'training_submitted': False}


def main(argv: list[str] | None = None) -> int:
    """Install an authorized code-only release; never submit training or modify Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--splits-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = install(args.archive, args.target, args.splits_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
