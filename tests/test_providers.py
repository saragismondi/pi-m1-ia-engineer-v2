"""Tests de los proveedores compatibles con OpenAI (Groq y OpenAI). Sin red ni LLM: costo 0.

Uso (sin dependencias extra):
    python tests/test_providers.py

Mejora pedida en el feedback de Henry (17/11/2026): un proveedor OpenAI real. Estos tests comprueban que
OpenAI se arma bien y que Groq sigue mandando exactamente los mismos parámetros que antes del cambio.
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python tests/test_providers.py`

from src.metrics import estimate_cost_usd  # noqa: E402
from src.providers import PROVIDERS, GroqProvider, OpenAIProvider, get_provider  # noqa: E402
from src.schemas import output_json_schema  # noqa: E402


@contextmanager
def env(**values):
    """Pone variables de entorno durante el test y deja las originales al terminar (None = borrarla)."""
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_openai_esta_registrado():
    assert "openai" in PROVIDERS


def test_openai_sin_clave_da_error_claro():
    with env(OPENAI_API_KEY=None):
        try:
            get_provider("openai")
        except ValueError as e:
            assert "OPENAI_API_KEY" in str(e)
        else:
            raise AssertionError("debía fallar sin clave")


def test_openai_usa_gpt_4o_mini_y_la_url_de_openai():
    with env(OPENAI_API_KEY="clave-falsa", OPENAI_MODEL=None):
        provider = get_provider("openai")
    assert isinstance(provider, OpenAIProvider) and provider.name == "openai"
    assert provider.model == "gpt-4o-mini"
    assert "api.openai.com" in str(provider.client.base_url)


def test_openai_no_manda_reasoning_effort():
    # gpt-4o-mini no razona: el parámetro no se manda.
    with env(OPENAI_API_KEY="clave-falsa"):
        req = get_provider("openai").build_request("sys", "user", output_json_schema())
    assert "reasoning_effort" not in req
    assert req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert req["temperature"] == 0.0 and req["max_tokens"] == int(os.getenv("MAX_TOKENS", "500"))


def test_groq_manda_lo_mismo_que_antes():
    # Mismos parámetros que el GroqProvider original: reasoning_effort=low, json_schema estricto, su URL.
    with env(GROQ_API_KEY="clave-falsa", GROQ_MODEL=None, GROQ_REASONING_EFFORT=None):
        provider = get_provider("groq")
    req = provider.build_request("sys", "user", output_json_schema())
    assert isinstance(provider, GroqProvider)
    assert req["model"] == "openai/gpt-oss-120b" and req["reasoning_effort"] == "low"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "api.groq.com" in str(provider.client.base_url)


def test_complete_lee_texto_y_tokens():
    with env(OPENAI_API_KEY="clave-falsa"):
        provider = get_provider("openai")
    fake = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=700, completion_tokens=200),
    )
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: fake)))
    result = provider.complete("sys", "user")
    assert (result.text, result.provider, result.prompt_tokens, result.completion_tokens) == ('{"ok": 1}', "openai", 700, 200)


def test_costo_de_gpt_4o_mini():
    # 1M de entrada a 0,15 + 1M de salida a 0,60 (precio oficial consultado el 17/11/2026).
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert estimate_cost_usd("gpt-4o-mini", 700, 200) == round(700 / 1e6 * 0.15 + 200 / 1e6 * 0.60, 8)


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FALLA {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
