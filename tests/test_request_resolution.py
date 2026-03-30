from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.api.endpoints.speech import _resolve_request_model_and_language
from app.models.requests import TTSRequest


class RequestValidationTests(unittest.TestCase):
    def test_request_accepts_language_only(self) -> None:
        request = TTSRequest(input="hello", language="sv", model="multilingual")
        self.assertEqual(request.language, "sv")
        self.assertIsNone(request.language_id)

    def test_request_accepts_language_id_only(self) -> None:
        request = TTSRequest(input="hello", language_id="de", model="multilingual")
        self.assertEqual(request.language_id, "de")
        self.assertIsNone(request.language)

    def test_request_rejects_invalid_model(self) -> None:
        with self.assertRaises(ValueError):
            TTSRequest(input="hello", model="bad-model")

    def test_conflicting_language_fields_raise_validation_error(self) -> None:
        with patch("app.api.endpoints.speech.is_multilingual", return_value=True):
            with patch("app.api.endpoints.speech.is_multilingual_runtime_ready", return_value=True):
                with self.assertRaises(HTTPException) as ctx:
                    _resolve_request_model_and_language("multilingual", "sv", "en")

        self.assertEqual(ctx.exception.status_code, 400)

    def test_multilingual_defaults_to_en(self) -> None:
        with patch("app.api.endpoints.speech.is_multilingual", return_value=True):
            with patch("app.api.endpoints.speech.is_multilingual_runtime_ready", return_value=True):
                model, language = _resolve_request_model_and_language("multilingual", None, None)
        self.assertEqual(model, "multilingual")
        self.assertEqual(language, "en")

    def test_turbo_forces_en_language(self) -> None:
        with patch("app.api.endpoints.speech.is_multilingual", return_value=True):
            with patch("app.api.endpoints.speech.is_multilingual_runtime_ready", return_value=True):
                model, language = _resolve_request_model_and_language("turbo", "sv", None)
        self.assertEqual(model, "turbo")
        self.assertEqual(language, "en")


if __name__ == "__main__":
    unittest.main()
