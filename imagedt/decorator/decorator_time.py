import time
from functools import wraps


def time_cost(func):
    @wraps(func)
    def _time_cost(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        print(
            f"==> time-cost: {time.perf_counter() - start:6f} (s)     {func.__name__} "
        )
        return ret

    return _time_cost
