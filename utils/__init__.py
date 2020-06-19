# msc
from utils.msc import logging, get_time, print_args
from utils.msc import expand_tensor
from utils.msc import save_checkpoint, load_checkpoint, load_replay_memory

# activation function
from utils.act_funcs import get_nonlinear_func

# stat
from utils.stat import get_covmat, sample_gaussian, logprob_gaussian

# jacobian clamping
from utils.jacobian_clamping import minrelu, jac_clamping_loss, cond_jac_clamping_loss
