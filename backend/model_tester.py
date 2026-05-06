"""Tester modeli — wysyła prompt przez OpenAI-compatible API."""

import json
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class TestResult:
    """Wynik testu modelu."""
    prompt: str
    response: Optional[str]
    error: Optional[str]
    tokens_used: Optional[int]
    time_taken: Optional[float]
    model_name: str
    endpoint: str


def test_model(
    endpoint: str,
    prompt: str,
    model_name: str = "default",
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 60,
) -> TestResult:
    """Wyślij prompt do API i zwróć wynik.

    Args:
        endpoint: URL serwera (np. http://127.0.0.1:8000)
        prompt: Tekst do wysłania
        model_name: Nazwa modelu (dla API)
        max_tokens: Maksymalna liczba tokenów w odpowiedzi
        temperature: Temperatura generowania
        timeout: Limit czasu w sekundach

    Returns:
        TestResult z odpowiedzią lub błędem
    """
    import time

    url = f"{endpoint.rstrip('/')}/v1/completions"
    start_time = time.time()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        elapsed = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()

            # Extrahuj tekst odpowiedzi
            if "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0].get("text", "")
            else:
                text = json.dumps(data, indent=2)

            # Extrahuj użycie tokenów
            tokens = None
            usage = data.get("usage", {})
            if usage:
                tokens = usage.get("completion_tokens") or usage.get("total_tokens")

            return TestResult(
                prompt=prompt,
                response=text,
                error=None,
                tokens_used=tokens,
                time_taken=elapsed,
                model_name=model_name,
                endpoint=endpoint,
            )
        else:
            return TestResult(
                prompt=prompt,
                response=None,
                error=f"HTTP {resp.status_code}: {resp.text}",
                tokens_used=None,
                time_taken=elapsed,
                model_name=model_name,
                endpoint=endpoint,
            )

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return TestResult(
            prompt=prompt,
            response=None,
            error=f"Timeout po {elapsed:.1f}s — serwer nie odpowiada",
            tokens_used=None,
            time_taken=elapsed,
            model_name=model_name,
            endpoint=endpoint,
        )
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start_time
        return TestResult(
            prompt=prompt,
            response=None,
            error=f"Błąd połączenia: {e}",
            tokens_used=None,
            time_taken=elapsed,
            model_name=model_name,
            endpoint=endpoint,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            prompt=prompt,
            response=None,
            error=str(e),
            tokens_used=None,
            time_taken=elapsed,
            model_name=model_name,
            endpoint=endpoint,
        )


def check_server_status(endpoint: str, timeout: int = 5) -> bool:
    """Sprawdź czy serwer odpowiada."""
    try:
        resp = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
