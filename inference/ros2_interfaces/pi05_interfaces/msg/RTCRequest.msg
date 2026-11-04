# Actual Piper state and unsent command targets at the replan control tick.
uint64 request_id
uint64 generation
uint64 observation_tick
builtin_interfaces/Time observation_stamp
builtin_interfaces/Time state_stamp
float64 captured_monotonic
uint32 delay_steps
uint32 overlap_count
float32[7] state
float32[] old_targets
