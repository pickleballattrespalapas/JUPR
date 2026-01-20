import time
from postgrest.exceptions import APIError

def sb_retry(fn, retries: int = 4, base_sleep: float = 0.6):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except APIError:
            raise
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** attempt))
    raise last
