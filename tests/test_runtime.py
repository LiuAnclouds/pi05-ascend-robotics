"""Runtime regression tests. Fake Piper only; never connect to CAN or cameras."""

import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from runtime.piper import send_motion_chunk
from runtime.report import RunReport


class FakePiper:
    """Record host target calls and return fresh healthy SDK-shaped feedback."""

    def __init__(self):
        """Initialize in-memory calls and state; no hardware resources are used."""
        self.calls, self.gripper_calls, self.times = [], [], []

    def GetArmStatus(self):
        """Return a normal CAN/MOVE_J feedback object."""
        return NS(time_stamp=time.time(), arm_status=NS(ctrl_mode=1, arm_status=0,
                  mode_feed=1, teach_status=0, err_code=0))

    def GetArmLowSpdInfoMsgs(self):
        """Return six enabled motors with no faults."""
        return NS(time_stamp=time.time(), **{f'motor_{i}': NS(foc_status_code=64) for i in range(1, 7)})

    def GetArmJointMsgs(self):
        """Return the latest target as simulated feedback, in SDK millidegrees."""
        values = self.calls[-1] if self.calls else [0] * 6
        return NS(time_stamp=time.time(), joint_state=NS(**{f'joint_{i + 1}': v for i, v in enumerate(values)}))

    def GetArmGripperMsgs(self):
        """Return simulated gripper feedback in thousandths of a millimetre."""
        return NS(time_stamp=time.time(), gripper_state=NS(grippers_angle=0))

    def JointCtrl(self, *values):
        """Save one target locally instead of transmitting a CAN message."""
        self.calls.append(values)
        self.times.append(time.perf_counter())

    def GripperCtrl(self, *values):
        """Save one gripper target locally; no physical effect."""
        self.gripper_calls.append(values)


class RuntimeTests(unittest.TestCase):
    """Check queue snapshots, finalization, failures and complete motion chunks."""

    def test_snapshot_and_interleaved_rounds(self):
        """Mutations cannot alter pending journal entries or confuse round IDs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            report = RunReport(path, {})
            record = {'iteration': 0, 'action_raw': [[1.5, 2.5]], 'execution': {}}
            report.event('prediction', record)
            record['action_raw'][0][0] = 99
            report.event('prediction', {'iteration': 1, 'action_raw': [[3.0]]})
            report.event('round_finished', {'iteration': 1, 'outcome': 'completed'})
            report.finish('interrupted', {})
            saved = json.loads(path.read_text())
            by_id = {r['iteration']: r for r in saved['rounds']}
            self.assertEqual(by_id[0]['action_raw'][0][0], 1.5)
            self.assertEqual(by_id[0]['outcome'], 'interrupted_before_execution_record')
            self.assertEqual(by_id[1]['outcome'], 'completed')
            self.assertFalse(report._writer.is_alive())
            self.assertFalse(path.with_suffix('.jsonl').exists())

    def test_disk_does_not_block_control_and_queue_is_bounded(self):
        """A blocked disk must not block event submission; overload is explicit."""
        with tempfile.TemporaryDirectory() as directory:
            report = RunReport(Path(directory) / 'result.json', {})
            report._queue.join()
            entered, release = threading.Event(), threading.Event()
            def blocked_fsync(_fd):
                """Hold the background writer until the test releases the fake disk."""
                entered.set()
                release.wait(5)
            with patch('runtime.report.os.fsync', side_effect=blocked_fsync):
                report.event('test', {})
                self.assertTrue(entered.wait(1))
                try:
                    for index in range(8):
                        report.event('test', {'index': index})
                    with self.assertRaisesRegex(RuntimeError, 'queue is full'):
                        report.event('test', {})
                finally:
                    release.set()
                    report._queue.join()
            report.finish('completed', {})

    def test_writer_failure_preserves_journal(self):
        """Storage errors propagate and do not erase the recoverable journal."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            report = RunReport(path, {})
            report._queue.join()
            with patch('runtime.report.os.fsync', side_effect=OSError('disk full')):
                report.event('test', {})
                report._queue.join()
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                report.check_writer()
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                report.finish('failed', {})
            self.assertFalse(report._writer.is_alive())
            self.assertTrue(path.with_suffix('.jsonl').exists())

    def test_cleanup_event_waits_for_queue_without_losing_records(self):
        """Shutdown backpressure preserves all events and interrupted status."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            report = RunReport(path, {})
            report._queue.join()
            entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
            errors = []

            def blocked_fsync(_fd):
                entered.set()
                release.wait(5)

            def submit_cleanup():
                try:
                    report.event('quick_stop', {'result': 'request_sent'}, wait=True)
                except Exception as error:
                    errors.append(error)
                finally:
                    submitted.set()

            with patch('runtime.report.os.fsync', side_effect=blocked_fsync):
                report.event('test', {})
                self.assertTrue(entered.wait(1))
                for index in range(8):
                    report.event('test', {'index': index})
                worker = threading.Thread(target=submit_cleanup)
                worker.start()
                try:
                    self.assertFalse(submitted.wait(0.15))
                finally:
                    release.set()
                    worker.join(2)
                    report._queue.join()
            self.assertTrue(submitted.is_set())
            self.assertEqual(errors, [])
            report.finish('interrupted', {})
            saved = json.loads(path.read_text())
            self.assertEqual(saved['status'], 'interrupted')
            self.assertEqual(len(saved['events']), 10)
            self.assertEqual(saved['events'][-1]['event'], 'quick_stop')

    def test_all_fifty_targets_unchanged(self):
        """The sender must emit all 50 points in order, including the gripper."""
        piper = FakePiper()
        trajectory = np.arange(350, dtype=np.float32).reshape(50, 7) / 10
        result = send_motion_chunk(piper, trajectory, 15, 1000, prepared=True)
        np.testing.assert_array_equal(piper.calls, np.rint(trajectory[:, :6] * 1000).astype(np.int32))
        self.assertEqual(result['points_sent'], 50)
        self.assertEqual(len(piper.gripper_calls), 50)
        self.assertIn('joint_timestamp', result['feedback_samples'][0])

    def test_stop_or_failure_prevents_further_targets(self):
        """Ctrl+C/background error must stop before any subsequent joint target."""
        for exception in (KeyboardInterrupt, RuntimeError):
            piper = FakePiper()
            def check():
                """Inject an interruption after exactly three target calls."""
                if len(piper.calls) == 3:
                    raise exception('stop')
            with self.assertRaises(exception):
                send_motion_chunk(piper, np.zeros((50, 7)), 15, 1000, prepared=True, check_pending=check)
            self.assertEqual(len(piper.calls), 3)

    def test_main_interruption_saves_partial_chunk_and_cleans_up(self):
        """Test the real entry's stop/report path using fake model and Piper only."""
        import contextlib
        import io
        from runtime import realtime_inference as entry

        class InterruptingPiper(FakePiper):
            """Simulate Ctrl+C on target four and record cleanup requests."""
            stopped = disconnected = False

            def JointCtrl(self, *values):
                """Interrupt before the fourth target is accepted."""
                if len(self.calls) == 3:
                    raise KeyboardInterrupt()
                super().JointCtrl(*values)

            def MotionCtrl_1(self, *values):
                """Record the software stop request without CAN transmission."""
                self.stopped = True

            def DisconnectPort(self):
                """Record connection cleanup."""
                self.disconnected = True

        class FakePolicy:
            """Produce a finite fixed 50-point chunk without allocating the NPU."""
            closed = False

            def predict_inputs(self, *_args, **_kwargs):
                """Return finite arrays and timing fields required by the real loop."""
                action = np.zeros((1, 50, 7), dtype=np.float32)
                return {'action_delta': action, 'action_target': action,
                        'action_normalized': np.zeros((1, 50, 32)), 'part1_ms': 0, 'part2_ms': [0] * 10}

            def close(self):
                """Record model cleanup."""
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            argv = ['realtime_inference.py', '--task', 'test', '--input', 'unused.pt',
                    '--send-motion', '--action-fps', '1000', '--iterations', '2', '--output', str(path)]
            piper, policy = InterruptingPiper(), FakePolicy()
            with patch('sys.argv', argv), patch.object(entry, 'connect_piper', return_value=piper), \
                    patch.object(entry, 'prepare_motion'), patch.object(entry, 'load_saved_inputs', return_value=[]), \
                    patch('runtime.official_om_policy.OfficialOMPolicy', return_value=policy), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(), 0)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['status'], 'interrupted')
            self.assertEqual(saved['rounds'][0]['motion_points_sent'], 3)
            self.assertEqual(saved['rounds'][0]['motion_horizon'], 50)
            self.assertTrue(piper.stopped and piper.disconnected and policy.closed)


if __name__ == '__main__':
    unittest.main()
