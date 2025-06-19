from prefect import task as prefect_task, flow as prefect_flow
from prefect.tasks import task_input_hash
from datetime import timedelta
from typing import Callable


def task(_func: Callable = None, **task_kwargs):
    def decorator(func):
        return prefect_task(
            persist_result=True,
            cache_key_fn=task_input_hash,
            cache_expiration=timedelta(days=2),
            retries=1,
            log_prints=True,
            **task_kwargs,
        )(func)

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def flow(_func: Callable = None, **flow_kwargs):
    DEFAULTS = {
        "log_prints": True,
        "retries": 1,
    }

    def decorator(func: Callable):
        return prefect_flow(**{**DEFAULTS, **flow_kwargs})(func)

    if _func is None:
        return decorator
    else:
        return decorator(_func)
