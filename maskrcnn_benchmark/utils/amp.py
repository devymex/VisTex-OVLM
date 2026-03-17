from contextlib import contextmanager
from functools import partial

@contextmanager
def nullcontext(enter_result=None, **kwargs):
    yield enter_result

try:
    from torch.cuda.amp import autocast, GradScaler
    from torch.amp import custom_fwd as _custom_fwd, custom_bwd as _custom_bwd
    custom_fwd = partial(_custom_fwd, device_type='cuda')
    custom_bwd = partial(_custom_bwd, device_type='cuda')
except:
    print('[Warning] Library for automatic mixed precision is not found, AMP is disabled!!')
    GradScaler = nullcontext
    autocast = nullcontext
    custom_fwd = nullcontext
    custom_bwd = nullcontext