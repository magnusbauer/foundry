import logging
import os

import torch
from beartype.claw import beartype_this_package
from environs import Env
from jaxtyping import install_import_hook


def _safe_bool_env(name: str, default: bool) -> tuple[bool, str | None]:
    """Parse a boolean env var without failing on empty or malformed values."""
    value = os.environ.get(name)
    if value is None:
        return default, None
    normalized = value.strip().lower()
    if normalized == "":
        return default, f"{name} is empty; using default {default!r}"
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True, None
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False, None
    return default, f"{name}={value!r} is not a valid boolean; using default {default!r}"


# Load environment variables from `.env` file
_env = Env()
_env.read_env()
should_typecheck, typecheck_warning = _safe_bool_env("TYPE_CHECK", default=False)
should_debug, debug_warning = _safe_bool_env("DEBUG", default=False)
should_check_nans, nan_warning = _safe_bool_env("NAN_CHECK", default=True)

# Set up logger
logger = logging.getLogger("foundry")
# ... set logging level based on `DEBUG` environment variable
logger.setLevel(logging.DEBUG if should_debug else logging.INFO)
# ... log the current mode
logger.debug("Debug mode: %s", should_debug)
logger.debug("Type checking mode: %s", should_typecheck)
logger.debug("NAN checking mode: %s", should_check_nans)
for warning in (typecheck_warning, debug_warning, nan_warning):
    if warning is not None:
        logger.warning(warning)

# Enable runtime type checking if `TYPE_CHECK` environment variable is set to `True`
if should_typecheck:
    beartype_this_package()
    install_import_hook("foundry", "beartype.beartype")

# Global flag for cuEquivariance availability
SHOULD_USE_CUEQUIVARIANCE = False

try:
    if torch.cuda.is_available():
        disable_cueq, cueq_warning = _safe_bool_env(
            "DISABLE_CUEQUIVARIANCE", default=False
        )
        if cueq_warning is not None:
            logger.warning(cueq_warning)
        if disable_cueq:
            logger.info("cuEquivariance usage disabled via DISABLE_CUEQUIVARIANCE")
        else:
            import cuequivariance_torch as cuet  # noqa: I001, F401

            SHOULD_USE_CUEQUIVARIANCE = True
            os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str(
                "CUEQ_DISABLE_AOT_TUNING", default="1"
            )
            os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str(
                "CUEQ_DEFAULT_CONFIG", default="1"
            )
            logger.info("cuEquivariance is available and will be used.")

except ImportError:
    logger.debug("cuEquivariance unavailable: import failed")


# Whether to disable checkpointing globally
DISABLE_CHECKPOINTING = False

# Export for easy access
__all__ = ["SHOULD_USE_CUEQUIVARIANCE", "DISABLE_CHECKPOINTING"]
