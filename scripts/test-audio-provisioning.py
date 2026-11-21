"""Offline fail-closed tests for the pinned audio installer."""
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('audio_provision', pathlib.Path(__file__).with_name('provision-audio-models.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ManifestTests(unittest.TestCase):
    def manifest(self):
        revision = 'a' * 40
        return {'schema_version': 1, 'revision': revision, 'repository': 'owner/model', 'files': [
            {'path': 'onnx/model.onnx', 'bytes': 123, 'sha256': 'b' * 64,
             'url': f'https://huggingface.co/owner/model/resolve/{revision}/onnx/model.onnx'}]}

    def test_valid_pinned_manifest(self):
        self.assertEqual(len(list(module.checked_entries(self.manifest()))), 1)

    def test_unsafe_paths_refused(self):
        for path in ('../model', '/model', 'C:/model', 'onnx/../../model', 'onnx\\model'):
            manifest = self.manifest()
            manifest['files'][0]['path'] = path
            with self.assertRaises(ValueError):
                list(module.checked_entries(manifest))

    def test_unpinned_or_wrong_url_refused(self):
        for url in ('http://huggingface.co/a', 'https://other.example/model',
                    'https://huggingface.co/owner/model/resolve/main/onnx/model.onnx'):
            manifest = self.manifest()
            manifest['files'][0]['url'] = url
            with self.assertRaises(ValueError):
                list(module.checked_entries(manifest))

    def test_bad_size_hash_and_duplicates_refused(self):
        for key, value in [('bytes', -1), ('bytes', True), ('sha256', 'incorrect')]:
            manifest = self.manifest()
            manifest['files'][0][key] = value
            with self.assertRaises(ValueError):
                list(module.checked_entries(manifest))
        manifest = self.manifest()
        manifest['files'] *= 2
        with self.assertRaises(ValueError):
            list(module.checked_entries(manifest))


if __name__ == '__main__':
    unittest.main()
