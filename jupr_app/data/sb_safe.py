import time

import httpx


def safe_execute(request_builder, retries: int = 3, delay: float = 0.4):
    """
    Wrap Supabase .execute() calls with retry logic to prevent transient
    HTTP/2 ReadError failures.
    """
    for attempt in range(retries):
        try:
            return request_builder.execute()
        except httpx.ReadError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
