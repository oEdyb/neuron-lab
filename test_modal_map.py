import unittest
from unittest.mock import patch

import modal_app


class ModalMapTests(unittest.TestCase):
    def test_gemma_command_timeout_and_private_token(self):
        secret = object()
        with patch('huggingface_hub.get_token', return_value='test-token'), \
             patch('modal_app.modal.Secret.from_dict', return_value=secret) as create_secret, \
             patch.object(modal_app.run, 'with_options') as options:
            options.return_value.remote.return_value = ({}, 0)
            modal_app.main('map.py', '--model google/gemma-4-E2B-it --out /cache/maps',
                           timeout=86400, hf_auth=True)
            create_secret.assert_called_once_with({'HF_TOKEN': 'test-token'})
            options.assert_called_once_with(timeout=86400, secrets=[secret])
            options.return_value.remote.assert_called_once_with(
                'map.py', ['--model', 'google/gemma-4-E2B-it', '--out', '/cache/maps'])

    def test_missing_auth_and_invalid_timeout_never_call_remote(self):
        with patch('huggingface_hub.get_token', return_value=None), \
             patch.object(modal_app.run, 'with_options') as options:
            with self.assertRaisesRegex(ValueError, 'token'):
                modal_app.main('map.py', hf_auth=True)
            with self.assertRaisesRegex(ValueError, 'timeout'):
                modal_app.main('map.py', timeout=86401)
            options.assert_not_called()


if __name__ == '__main__':
    unittest.main()
