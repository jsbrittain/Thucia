import logging
from prefect import get_run_logger
from prefect.context import FlowRunContext, TaskRunContext


class PrefectLogRedirectHandler(logging.Handler):
    def emit(self, record):
        # Prevent recursion and shutdown logging loops
        if record.name.startswith("prefect") or record.name == "httpx":
            return

        # Avoid emitting after shutdown begins
        try:
            if not (TaskRunContext.get() or FlowRunContext.get()):
                return
        except Exception:
            return  # likely already torn down

        try:
            message = self.format(record)
            get_run_logger().log(record.levelno, message)
        except Exception:
            pass  # prevent cascading failure at shutdown


def enable_prefect_logging_redirect(level=logging.INFO):
    """
    Redirect standard logging to Prefect's logger if in a flow/task.
    """
    handler = PrefectLogRedirectHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(min(root_logger.level, level))
