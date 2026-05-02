from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ServiceResult:
    ok: bool
    data: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    @classmethod
    def success(
        cls,
        data: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> "ServiceResult":
        return cls(ok=True, data=data or {}, warnings=warnings or [], errors=[])

    @classmethod
    def failure(
        cls,
        errors: list[str] | str,
        data: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> "ServiceResult":
        error_list = [errors] if isinstance(errors, str) else errors
        return cls(ok=False, data=data or {}, warnings=warnings or [], errors=error_list)
