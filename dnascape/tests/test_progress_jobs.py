import contextlib
import time
import unittest
from io import StringIO
from threading import Event

from app_streamlit.jobs import (
    cancel_background_job,
    poll_background_job,
    start_background_job,
)
from dnascape.utils import etaf, progress_context


class ProgressContextTests(unittest.TestCase):
    def test_eta_progress_is_thread_local_and_can_suppress_console_output(self):
        updates = []
        output = StringIO()

        with contextlib.redirect_stdout(output):
            with progress_context(
                lambda value, text: updates.append((value, text)),
                suppress_output=True,
            ):
                etaf(1, 2, time.time() - 1.0, label="rsim")

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(updates), 1)
        self.assertAlmostEqual(updates[0][0], 0.49)
        self.assertIn("Simulating", updates[0][1])


class BackgroundJobTests(unittest.TestCase):
    @staticmethod
    def _poll_until_terminal(job, timeout=2.0):
        deadline = time.time() + timeout
        while job.running and time.time() < deadline:
            poll_background_job(job)
            time.sleep(0.002)
        poll_background_job(job)
        return job

    def test_progress_and_result_are_drained_without_blocking(self):
        def worker(*, job_context):
            job_context.report(0.4, "Step one")
            job_context.report(0.8, "Step two")
            return {"ok": True}

        job = start_background_job("test", "signature", worker)
        self._poll_until_terminal(job)

        self.assertEqual(job.status, "complete")
        self.assertEqual(job.progress_value, 1.0)
        self.assertEqual(job.result, {"ok": True})

    def test_cancellation_is_acknowledged_cooperatively(self):
        started = Event()

        def worker(*, job_context):
            started.set()
            while True:
                job_context.raise_if_cancelled()
                time.sleep(0.002)

        job = start_background_job("test", "signature", worker)
        self.assertTrue(started.wait(timeout=1.0))
        self.assertTrue(cancel_background_job(job))
        self._poll_until_terminal(job)

        self.assertEqual(job.status, "cancelled")
        self.assertTrue(job.cancellation_requested)


if __name__ == "__main__":
    unittest.main()
