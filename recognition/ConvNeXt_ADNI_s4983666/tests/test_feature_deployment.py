"""Synthetic verification of the code-only deployment boundary."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from slurm.deploy_feature_source import install, verify_bundle


class DeploymentTests(unittest.TestCase):
    """Reject malicious or corrupt members and preserve unrelated server files."""

    def bundle(self, base: Path, payload: dict, hashes: dict | None = None) -> Path:
        """Create an explicitly synthetic source bundle."""
        archive = base/'code.tar.gz'
        contents = payload | {'BUNDLE_MANIFEST.json':json.dumps({'sha256':hashes or {
            name:hashlib.sha256(value).hexdigest() for name,value in payload.items()}}).encode()}
        with tarfile.open(archive,'w:gz') as stream:
            for name, value in contents.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                stream.addfile(info,io.BytesIO(value))
        return archive

    def test_path_traversal_and_results_are_never_installable(self) -> None:
        """Code bundles cannot smuggle patient data, run logs or credentials."""
        for name in ('../train.py','/tmp/train.py','runs/config.json','outputs/report.json','best.pt','.env','AGENTS.md'):
            with tempfile.TemporaryDirectory() as temp:
                archive = self.bundle(Path(temp),{name:b'x'})
                with self.assertRaises(ValueError):
                    verify_bundle(archive)

    def test_corrupt_member_is_rejected_before_any_write(self) -> None:
        """Every source byte must match the declared digest."""
        with tempfile.TemporaryDirectory() as temp:
            archive = self.bundle(Path(temp),{'train.py':b'x'},{'train.py':'0'*64})
            with self.assertRaisesRegex(ValueError,'checksum'):
                verify_bundle(archive)

    def test_install_backs_up_only_source_and_preserves_runs_and_manifests(self) -> None:
        """Existing experiments and unrelated files stay byte-identical."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            target, splits = base/'code',base/'splits'
            target.mkdir(); splits.mkdir()
            (target/'train.py').write_bytes(b'old')
            (target/'runs').mkdir()
            (target/'runs/metrics.json').write_bytes(b'experiment')
            (splits/'train.csv').write_bytes(b'protected')
            archive = self.bundle(base,{'train.py':b'new','engine/new.py':b'newcode'})
            with mock.patch('slurm.deploy_feature_source.frozen_seals',return_value={'COMPLETED.json':'test'}):
                result = install(archive,target,splits,check_jobs=False)
            self.assertEqual((target/'train.py').read_bytes(),b'new')
            self.assertEqual((target/'runs/metrics.json').read_bytes(),b'experiment')
            self.assertEqual((splits/'train.csv').read_bytes(),b'protected')
            with tarfile.open(result['backup']) as backup:
                self.assertEqual(backup.getnames(),['train.py'])
                self.assertEqual(backup.extractfile('train.py').read(),b'old')

    def test_symlink_destination_cannot_redirect_installation(self) -> None:
        """An existing symlink cannot point a source write into another directory."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            target, splits, protected = base/'code',base/'splits',base/'protected'
            target.mkdir(); splits.mkdir(); protected.mkdir()
            (target/'engine').symlink_to(protected,target_is_directory=True)
            archive = self.bundle(base,{'engine/new.py':b'new'})
            with mock.patch('slurm.deploy_feature_source.frozen_seals',return_value={'COMPLETED.json':'test'}):
                with self.assertRaisesRegex(ValueError,'Symlink'):
                    install(archive,target,splits,check_jobs=False)
            self.assertFalse((protected/'new.py').exists())

    def test_mid_install_failure_rolls_back_replaced_source(self) -> None:
        """A failed deployment restores replaced code without deleting experiment files."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            target, splits = base/'code',base/'splits'
            target.mkdir(); splits.mkdir()
            (target/'train.py').write_bytes(b'old')
            archive = self.bundle(base,{'train.py':b'new','engine/new.py':b'new'})
            real_replace = Path.replace
            def fail_second(path: Path, destination: Path) -> Path:
                """Inject a filesystem error after the first installed source file."""
                if destination.name == 'new.py':
                    raise OSError('synthetic deployment failure')
                return real_replace(path,destination)
            with mock.patch('slurm.deploy_feature_source.frozen_seals',return_value={'COMPLETED.json':'test'}),mock.patch.object(Path,'replace',fail_second):
                with self.assertRaises(OSError):
                    install(archive,target,splits,check_jobs=False)
            self.assertEqual((target/'train.py').read_bytes(),b'old')
            self.assertFalse((target/'engine/new.py').exists())
