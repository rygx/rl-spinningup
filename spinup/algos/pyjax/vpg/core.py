import jax
import jax.numpy as jnp
import jax.nn as nn
from jax.random import categorical, normal

import scipy.signal
from gymnasium.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if jnp.isscalar(shape) else (length, *shape)

# def placeholder(dim=None):
#     return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

# def placeholders(*args):
#     return [placeholder(dim) for dim in args]

# def placeholder_from_space(space):
#     if isinstance(space, Box):
#         return placeholder(space.shape)
#     elif isinstance(space, Discrete):
#         return tf.placeholder(dtype=tf.int32, shape=(None,))
#     raise NotImplementedError

# def placeholders_from_spaces(*args):
#     return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=jnp.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = jax.nn.dense(x, h, activation=activation)
    return jax.nn.dense(x, hidden_sizes[-1], activation=output_activation)

def count_vars(module):
    params = jax.tree_util.tree_leaves(module.init(jnp.zeros((1,))))
    return sum(jnp.prod(p.shape) for p in params)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(jnp.exp(log_std)+EPS))**2 + 2*log_std + jnp.log(2*jnp.pi))
    return jnp.sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = jax.nn.log_softmax(logits)
    pi = categorical(jax.random.PRNGKey(0), logits, shape=())
    logp = jnp.sum(jnp.eye(act_dim)[a] * logp_all, axis=1)
    logp_pi = jnp.sum(jnp.eye(act_dim)[pi] * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = jnp.array(-0.5 * jnp.ones(act_dim, dtype=jnp.float32))
    std = jnp.exp(log_std)
    pi = mu + normal(jax.random.PRNGKey(0), shape=mu.shape) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=jnp.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    v = jnp.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v
