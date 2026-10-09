"""Runtime regression tests. Fake Piper only; never connect to CAN or cameras."""

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from runtime.piper import send_motion_chunk, smooth_motion_trajectory, prepare_motion, MotionNotReady
from runtime.prompt import PromptController, send_prompt, send_continue
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

    def test_joint_filter_preserves_horizon_gripper_and_bounds(self):
        """Filter only joints and bound changes from the current state onward."""
        trajectory = np.zeros((50, 7), dtype=np.float32)
        trajectory[:, :6] = np.linspace(0, 40, 50)[:, None]
        trajectory[:, 6] = np.arange(50)
        before = trajectory.copy()
        state = np.zeros(7, dtype=np.float32)
        sent = smooth_motion_trajectory(trajectory, state, state)
        self.assertEqual(sent.shape, (50, 7))
        np.testing.assert_array_equal(sent[:, 6], trajectory[:, 6])
        np.testing.assert_array_equal(trajectory, before)
        steps = np.diff(np.vstack((state[:6], sent[:, :6])), axis=0)
        self.assertLessEqual(np.max(np.abs(steps)), 1.20001)
        self.assertTrue(np.isfinite(sent).all())

    def test_filter_reduces_variation_and_preserves_slow_forward_motion(self):
        """Deadband does not suppress small deliberate forward movement."""
        state = np.zeros(7, dtype=np.float32)
        trajectory = np.zeros((50, 7), dtype=np.float32)
        trajectory[:, 0] = np.tile([0.2, 0.0], 25)
        baseline = smooth_motion_trajectory(trajectory, state, state,
                                             filter_alpha=1, reversal_deadband_deg=0)
        sent = smooth_motion_trajectory(trajectory, state, state)
        self.assertLess(np.abs(np.diff(sent[:, 0])).sum(),
                        np.abs(np.diff(baseline[:, 0])).sum())
        trajectory[:, 0] = np.linspace(0.01, 0.5, 50)
        sent = smooth_motion_trajectory(trajectory, state, state)
        self.assertGreater(sent[-1, 0], 0.48)
        self.assertTrue((np.diff(sent[:, 0]) >= 0).all())

    def test_prompt_reader_latest_request_and_cleanup(self):
        """A separate process sends tasks, rejects blanks and survives reconnection."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompt.sock'
            controller = PromptController(path)
            try:
                with self.assertRaises(ValueError):
                    send_prompt('  ', path)
                self.assertIsNone(controller.take())
                send_prompt('old task', path)
                send_prompt('/prompt new task', path)
                self.assertEqual(controller.take(), 'new task')
                self.assertIsNone(controller.take())
                client = Path(__file__).resolve().parents[1] / 'runtime/set_prompt.py'
                result = subprocess.run([sys.executable, str(client), '--socket', str(path),
                                         '--task', 'Pick up the pen'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Queued:', result.stdout)
                self.assertEqual(controller.take(), 'Pick up the pen')
                with self.assertRaisesRegex(RuntimeError, 'Another inference'):
                    PromptController(path)
                send_prompt('still connected', path)
                self.assertEqual(controller.take(), 'still connected')
            finally:
                controller.close()
            self.assertFalse(controller._thread.is_alive())
            self.assertFalse(path.exists())
            with self.assertRaises(OSError):
                send_prompt('no receiver', path)
            path.touch()  # Simulate the socket left after an unclean exit.
            restarted = PromptController(path)
            restarted.close()

    def test_reset_explicitly_disables_before_enabling(self):
        """Even if reset leaves torque enabled, explicit disable precedes enable."""
        class StartupPiper(FakePiper):
            def __init__(self):
                super().__init__()
                self.enabled, self.mode, self.arm = True, 0, 1
                self.transitions = []

            def GetArmStatus(self):
                return NS(time_stamp=time.time(), arm_status=NS(ctrl_mode=self.mode,
                          arm_status=self.arm, mode_feed=1, teach_status=0, err_code=0))

            def GetArmLowSpdInfoMsgs(self):
                return NS(time_stamp=time.time(), **{f'motor_{i}': NS(
                    foc_status_code=64 if self.enabled else 0) for i in range(1, 7)})

            def MotionCtrl_1(self, *_args):
                self.transitions.append('reset')
                self.arm = 0

            def DisablePiper(self):
                self.transitions.append('disable')
                self.enabled = False

            def EnablePiper(self):
                self.transitions.append('enable')
                self.enabled = True

            def ModeCtrl(self, *_args):
                self.transitions.append('mode')
                self.mode = 1

        piper = StartupPiper()
        result = prepare_motion(piper, 10)
        self.assertEqual(piper.transitions, ['reset', 'disable', 'enable', 'mode'])
        self.assertEqual(result['after']['control_mode'], 1)
        self.assertEqual(piper.calls, [])

    def test_w_gate_initial_and_after_task_change(self):
        """Initial and replacement tasks both require W at a waiting boundary."""
        import contextlib
        import io
        from runtime.realtime_inference import wait_for_task

        with tempfile.TemporaryDirectory() as directory:
            controller = PromptController(Path(directory) / 'prompt.sock')
            events, results = [], []
            report = NS(check_writer=lambda: None,
                        event=lambda name, data: events.append((name, data)))

            def wait_until(predicate):
                deadline = time.monotonic() + 2
                while not predicate() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(predicate())

            def run_gate(task):
                results.append(wait_for_task(controller, task, report, len(results)))

            worker = None
            try:
                with self.assertRaisesRegex(ValueError, 'Not waiting yet'):
                    send_continue(controller.path)
                with contextlib.redirect_stdout(io.StringIO()):
                    worker = threading.Thread(target=run_gate, args=('initial task',))
                    worker.start()
                    wait_until(lambda: controller._waiting)
                    time.sleep(0.1)
                    self.assertTrue(worker.is_alive())
                    self.assertEqual(results, [])
                    send_continue(controller.path)
                    worker.join(2)
                    self.assertEqual(results, ['initial task'])
                    with self.assertRaises(ValueError):
                        send_continue(controller.path)  # No future permission queued.
                    self.assertEqual(controller.poll_gate(), (None, True))
                    send_prompt('replacement task', controller.path)
                    with self.assertRaises(ValueError):
                        send_continue(controller.path)  # Must first reach the boundary.
                    worker = threading.Thread(target=run_gate, args=('initial task',))
                    worker.start()
                    wait_until(lambda: controller._waiting)
                    time.sleep(0.1)
                    self.assertTrue(worker.is_alive())
                    self.assertEqual(results, ['initial task'])
                    send_continue(controller.path)
                    worker.join(2)
                    self.assertEqual(results, ['initial task', 'replacement task'])
                self.assertEqual([name for name, _ in events].count('operator_continued'), 2)
                self.assertIn('task_changed', [name for name, _ in events])
            finally:
                if worker is not None and worker.is_alive():
                    controller.continue_task()
                    worker.join(2)
                controller.close()

    def test_w_wait_interrupt_propagates(self):
        """Ctrl+C while waiting reaches the existing stop/cleanup handler."""
        from runtime.realtime_inference import wait_for_task
        controller = NS(poll_gate=lambda: (None, False))
        report = NS(check_writer=lambda: None, event=lambda *_args: None)
        with patch('runtime.realtime_inference.time.sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                wait_for_task(controller, 'test', report, 0)

    def test_live_task_switch_updates_model_input_and_report(self):
        """A boundary change reaches both tokenizer input and per-round records."""
        import contextlib
        import io
        from runtime import realtime_inference as entry

        tasks = []
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        def make_inputs(_front, _wrist, _state, task):
            tasks.append(task)
            return task

        def predict(*_args, **_kwargs):
            # Allow the real background journal to drain, as during OM inference.
            time.sleep(0.03)
            action = np.zeros((1, 50, 7), dtype=np.float32)
            return {'action_delta': action, 'action_target': action,
                    'action_normalized': action, 'part1_ms': 0, 'part2_ms': [0] * 10}

        camera = NS(read=lambda: (frame, frame, 0), metadata=lambda: {}, close=lambda: None)
        policy = NS(make_inputs=make_inputs, predict_inputs=predict, close=lambda: None)
        piper = FakePiper()
        piper.DisconnectPort = lambda: None
        requests = iter([(None, False, 1), (None, True, 1),
                         ('Pick up the pen', False, 2), (None, True, 2)])

        def poll_gate():
            task, running, expected_input_count = next(requests)
            self.assertEqual(len(tasks), expected_input_count,
                             'No new model input/prediction may be created while waiting for W')
            return task, running

        controller = NS(poll_gate=poll_gate, close=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            argv = ['realtime_inference.py', '--task', 'Pick up the can',
                    '--iterations', '2', '--output', str(path)]
            for option in ('--part1-om', '--part2-om', '--tokenizer', '--stats'):
                argv.extend([option, __file__])  # FakePolicy does not open assets.
            with patch('sys.argv', argv), patch('sys.stdin.isatty', return_value=False), \
                    patch.object(entry, 'CameraPair', return_value=camera), \
                    patch.object(entry, 'connect_piper', return_value=piper), \
                    patch.object(entry, 'PromptController', return_value=controller), \
                    patch('runtime.official_om_policy.OfficialOMPolicy', return_value=policy), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = entry.main()
                self.assertEqual(result, 0, json.loads(path.read_text()).get('summary'))
            saved = json.loads(path.read_text())
            self.assertEqual(tasks, ['Pick up the can', 'Pick up the can', 'Pick up the pen'])
            self.assertEqual([r['task'] for r in saved['rounds']], tasks[1:])
            changes = [e for e in saved['events'] if e['event'] == 'task_changed']
            self.assertEqual(changes[0]['data']['iteration'], 1)
            self.assertEqual(sum(e['event'] == 'waiting_for_operator' for e in saved['events']), 2)
            self.assertEqual(sum(e['event'] == 'operator_continued' for e in saved['events']), 2)

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
            for option in ('--part1-om', '--part2-om', '--tokenizer', '--stats'):
                argv.extend([option, __file__])  # FakePolicy does not open assets.
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
