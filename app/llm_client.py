import requests
import time

from config import LLM_URL


class LLMClient:
    def __init__(self):
        self.base_url = LLM_URL
        self.completion_url = f"{self.base_url}/completion"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list | None = None,
        retries: int = 6,
        retry_delay_seconds: float = 5.0,
    ) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or ["\n\nQuestion:", "###"],
            "stream": False,
        }

        last_exception = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.completion_url,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["content"].strip()
            except requests.HTTPError as exc:
                # llama-server returns 503 while loading model tensors.
                status_code = exc.response.status_code if exc.response is not None else None
                last_exception = exc
                if status_code == 503 and attempt < retries:
                    time.sleep(retry_delay_seconds)
                    continue
                raise
            except requests.RequestException as exc:
                last_exception = exc
                if attempt < retries:
                    time.sleep(retry_delay_seconds)
                    continue
                raise

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected error while generating completion")

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
