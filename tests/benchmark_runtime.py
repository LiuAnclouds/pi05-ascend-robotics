"""Board-only host-copy/device-cache comparison on saved real inputs; NO CAN.

Run from the project root with the Ascend environment loaded:
    python -m tests.benchmark_runtime
The raw measurements and equality checks are saved under outputs/runs.
"""

import json
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import torch

from include.project_paths import DEFAULT_PART1_INPUT, DEFAULT_RUN_DIR
from include.runtime_defs import DEFAULT_PART1_OM, DEFAULT_PART2_OM, DEFAULT_STATS, DEFAULT_TOKENIZER
from runtime.input_data import load_saved_inputs
from runtime.official_om_policy import OfficialOMPolicy
from runtime.piper import send_motion_chunk
from runtime.report import RunReport, pretty_json
from tests.test_runtime import FakePiper


def main():
    """Measure alternating paths and simulated 50-point sending; assert exact actions."""
    directory = DEFAULT_RUN_DIR / f'Runtime_benchmark_{datetime.now():%Y%m%d_%H%M%S}'
    directory.mkdir(parents=True)
    inputs = load_saved_inputs(DEFAULT_PART1_INPUT)
    metadata = torch.load(DEFAULT_PART1_INPUT, map_location='cpu', weights_only=False)['meta']
    state = np.asarray(metadata['state_raw'], dtype=np.float32)
    policy = OfficialOMPolicy(DEFAULT_PART1_OM, DEFAULT_PART2_OM, DEFAULT_TOKENIZER, DEFAULT_STATS, 10)
    cache = policy.device_cache
    results = {'hardware_motion': False, 'input': str(DEFAULT_PART1_INPUT), 'metadata': metadata,
               'cache_bytes': sum(buffer['size'] for buffer in cache.buffers), 'pairs': []}
    try:
        for mode in (None, cache):
            policy.device_cache = mode
            policy.predict_inputs(inputs, state=state, return_cache=False)
        for seed in range(10):
            pair, actions = {'seed': seed}, {}
            modes = [('host_copy', None), ('device_cache', cache)]
            if seed % 2:
                modes.reverse()
            for name, mode in modes:
                policy.device_cache = mode
                started = time.perf_counter()
                output = policy.predict_inputs(inputs, seed=seed, state=state, return_cache=False)
                pair[name] = {'total_ms': (time.perf_counter() - started) * 1000,
                              'part1_ms': output['part1_ms'], 'part2_ms': output['part2_ms']}
                actions[name] = {key: output[key].copy() for key in ('action_normalized', 'action_delta', 'action_target')}
            for key in actions['host_copy']:
                np.testing.assert_array_equal(actions['host_copy'][key], actions['device_cache'][key])
                assert np.isfinite(actions['device_cache'][key]).all()
            pair['exactly_equal'] = True
            results['pairs'].append(pair)
            print(json.dumps(pair), flush=True)
        # Changed observations must overwrite borrowed device buffers; verify
        # this independently from seed changes and the normal action-only path.
        changed = [value.copy() for value in inputs]
        changed[0], changed[1] = changed[1], changed[0]
        caches = {}
        for name, mode in [('host_copy', None), ('device_cache', cache)]:
            policy.device_cache = mode
            caches[name] = policy.predict_inputs(changed, seed=123, state=state, return_cache=True)
        for key in ('past_kv', 'prefix_pad_masks', 'action_normalized', 'action_target'):
            np.testing.assert_array_equal(caches['host_copy'][key], caches['device_cache'][key])
        results['changed_observation_cache_equal'] = True
        del caches
        # Exercise the original sender and asynchronous report with real model
        # actions, but a FakePiper that cannot transmit to any physical device.
        report = RunReport(directory / 'simulated_send.json', {'fake_piper': True})
        fake, rounds, previous = FakePiper(), [], None
        policy.device_cache = cache
        for index in range(4):
            started = time.perf_counter()
            output = policy.predict_inputs(inputs, seed=index, state=state, return_cache=False)
            infer_ms = (time.perf_counter() - started) * 1000
            record = {'iteration': index, 'action_raw': output['action_target'].tolist(),
                      'action_normalized': output['action_normalized'].tolist(), 'execution': {}}
            started = time.perf_counter()
            report.event('prediction', record)
            enqueue_ms = (time.perf_counter() - started) * 1000
            execution = send_motion_chunk(fake, output['action_target'][0], 15, 30, prepared=True,
                                           progress=record['execution'], check_pending=report.check_writer)
            record.update(outcome='completed', infer_ms=infer_ms, journal_enqueue_ms=enqueue_ms,
                          gap_ms=None if previous is None else (execution['first_command_time'] - previous) * 1000)
            previous = execution['last_command_time']
            report.event('round_finished', record)
            rounds.append({k: record[k] for k in ('iteration', 'infer_ms', 'journal_enqueue_ms', 'gap_ms')})
        report.finish('completed', {'rounds': 4, 'joint_calls': len(fake.calls)})
        assert len(fake.calls) == 200
        results['simulated_send'] = rounds
        results['summary'] = {mode: {'mean_ms': float(np.mean([p[mode]['total_ms'] for p in results['pairs']])),
                                     'p95_ms': float(np.percentile([p[mode]['total_ms'] for p in results['pairs']], 95))}
                              for mode in ('host_copy', 'device_cache')}
        (directory / 'comparison.json').write_text(pretty_json(results) + '\n', encoding='utf-8')
        print(pretty_json({'output': str(directory), 'summary': results['summary'], 'simulated_send': rounds}), flush=True)
    finally:
        cache = None
        policy.close()


if __name__ == '__main__':
    main()
