"""Session-scoped display of core logs without coupling training to Streamlit."""

from collections import deque
from contextlib import contextmanager
import logging
import threading
import time

import streamlit as st

_lock = threading.Lock()
_active = 0
_previous_level = logging.NOTSET


class RunLogHandler(logging.Handler):
    """Bounded log tail; never send another session's messages to this UI."""

    def __init__(self, render):
        super().__init__(logging.INFO)
        self.owner = threading.get_ident()
        self.render = render
        self.lines = deque(maxlen=80)
        self.last_render = 0.0
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        if record.thread != self.owner:
            return
        self.lines.append(self.format(record))
        now = time.monotonic()
        # Epochs can arrive faster than the browser can render. Stage and model
        # transitions remain immediate, and all records still reach CLI handlers.
        if "Epoch " not in record.getMessage() or now - self.last_render >= 0.5:
            self.flush()
            self.last_render = now

    def flush(self):
        if self.lines and threading.get_ident() == self.owner:
            self.render("\n".join(self.lines))


@contextmanager
def run_progress(label):
    """Display an operation's core logs in this Streamlit session.

    Use as a context manager or decorator around a synchronous GUI action.
    Only records from the calling thread are shown; Ray trial callbacks report
    from its driver. Nested/concurrent runs share logger-level bookkeeping and
    remove their own handlers in finally. Exceptions propagate to the caller
    after marking the status unsuccessful. The panel retains 80 records and
    throttles epoch rendering, without changing the underlying CLI log stream.
    """
    global _active, _previous_level
    logger = logging.getLogger("polynet")
    with st.status(label, expanded=True) as status:
        st.caption(
            "Live activity (latest 80 messages). Some fitting and tuning steps may "
            "run for a while between messages."
        )
        output = st.empty()
        handler = RunLogHandler(lambda text: output.code(text, language=None))
        with _lock:
            if _active == 0:
                _previous_level = logger.level
                logger.setLevel(min(logger.getEffectiveLevel(), logging.INFO))
            _active += 1
            logger.addHandler(handler)
        try:
            logger.info("%s started", label)
            yield
        except BaseException:
            status.update(label=f"{label} did not finish", state="error", expanded=True)
            raise
        else:
            logger.info("%s completed", label)
            status.update(label=f"{label} completed", state="complete")
        finally:
            with _lock:
                logger.removeHandler(handler)
                _active -= 1
                if _active == 0:
                    logger.setLevel(_previous_level)
            handler.flush()
            handler.close()
