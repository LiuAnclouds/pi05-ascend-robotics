# Complete model prediction anchored at the RTC request's observation tick.
uint64 request_id
uint64 generation
uint64 observation_tick
builtin_interfaces/Time observation_stamp
bool cancelled
string reason
float32[] targets
