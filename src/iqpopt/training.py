import time
from typing import Callable
from inspect import signature

import jax
import jax.numpy as jnp
import jaxopt
import optax
from tqdm import tqdm


class Trainer:
    """ Class that creates an Trainer object for loss callable."""
    def __init__(self, optimizer: str, loss: Callable, stepsize: float,
                 opt_jit: bool = False):
        """Creates the Trainer object.

        Args:
            optimizer (str): The string name of the jaxopt optimizer.
            loss (Callable): The loss function we want to optimize.
            stepsize (float): The stepsize that the optimizer will take.
            opt_jit (bool, optional): It jits the whole training process if True. Can't be used with functions
                that are not jittable without static argnames (mmd_loss_iqp being one example).Defaults to False.
        """

        key_exists = dict(signature(loss).parameters).pop("key", False)

        self.loss = loss if key_exists else lambda params, key, **kwargs: loss(
            params, **kwargs)
        self.optimizer = optimizer
        self.stepsize = stepsize

        if self.optimizer == "GradientDescent":
            self.opt = jaxopt.GradientDescent(
                self.loss, stepsize=self.stepsize, verbose=False, jit=opt_jit)
        elif self.optimizer == "Adam":
            self.opt = jaxopt.OptaxSolver(self.loss, optax.adam(
                self.stepsize), verbose=False, jit=opt_jit)
        elif self.optimizer == "BFGS":
            self.opt = jaxopt.BFGS(self.loss, verbose=False, jit=opt_jit)

    def train(self, n_iters: int, loss_kwargs: dict, val_kwargs: dict = None,
              convergence_interval: int = None, monitor_interval: int = None,
              turbo=None, random_state=666):
        """Training loop that minimizes the loss function with the given loss_kwargs.

        Args:
            n_iters (int): Number of iterations in the loop.
            loss_kwargs (dict): dictionary with all the arguments of the loss function with their values used for training.
                must contain a key 'params' that stores the training parameters
            val_kwargs (dict): dictionary with all the arguments except 'params' of the loss function with their values used for
                validation/convergence. if None, the training loss is used to decide convergence.
            convergence_interval (int): number of steps over which to decide convergence. If the loss does not decrease
                or increased over this scale, the training is stopped.
            monitor_interval (int, optional): Every "monitor_interval" iterations, the program saves the variable params
                in a historic list. Defaults to None, meaning there's no monitoring of the history of params.
            turbo (int or None, optional): If set to an int, jax.lax is used to compile every turbo number of training
                steps for faster training. If None, a standard for loop is used.
            random_state (int): seed for random functions
        """

        start = time.time()
        self.n_iters = n_iters
        self.losses = jnp.array([])
        self.val_losses = jnp.array([])

        fixed_loss_kwargs = loss_kwargs.copy()
        validation = True if val_kwargs is not None else False
        fixed_val_kwargs = val_kwargs.copy() if validation else {}
        params_init = fixed_loss_kwargs.pop("params")
        key = jax.random.PRNGKey(random_state)
        key1, key2 = jax.random.split(key, 2)
        key = fixed_loss_kwargs.pop("key", key1)
        key_val = fixed_val_kwargs.pop("key", key2)

        state = self.opt.init_state(params_init, **fixed_loss_kwargs, key=key)
        params = params_init
        if monitor_interval is not None:
            self.params_hist = jnp.empty((0, len(params_init)), float)
        converged = False

        def update(carry, x):
            """
            function used in jax.lax.scan to perform the update step
            """
            params = carry[0]
            state = carry[1]
            key = carry[2]
            key_val = carry[3]
            key1, key2 = jax.random.split(key, 2)
            key1_val, key2_val = jax.random.split(key_val, 2)

            params, state = self.opt.update(params, state, **fixed_loss_kwargs, key=key1)
            v = self.loss(params, **fixed_val_kwargs, key=key1_val) if validation else 0.

            if monitor_interval is not None:
                params_hist = jnp.where(x % monitor_interval == 0, params, 0)
            else:
                params_hist = None

            if self.optimizer == "GradientDescent":
                l = self.loss(params, **fixed_loss_kwargs, key=key1)
            else:
                l = state.value
            return [params, state, key2, key2_val], [l, v, params_hist]

        if turbo is not None:
            update = jax.jit(update)
            # use lax.scan to optimize every turbo steps. Speeds things up a bit.
            n_loops = n_iters//turbo
            n_loops = 1 if n_loops == 0 else n_loops
            carry = [params, state, key, key_val]
            with tqdm(total=n_iters, desc="Training Progress",
                      postfix={"loss": 0.0, "elapsed time": 0.0, "total time": 0.0}) as pbar:
                for loop in range(1, n_loops+1):
                    start2 = time.time()
                    carry, [losses, vals, params_hist] = jax.lax.scan(
                        update, carry, jnp.arange(turbo))
                    self.losses = jnp.concatenate((self.losses, losses))
                    self.val_losses = jnp.concatenate((self.val_losses, vals))
                    if monitor_interval is not None:
                        self.params_hist = jnp.concatenate(
                            (self.params_hist, params_hist))
                    else:
                        self.params_hist = []

                    pbar.set_postfix({"loss": f"{losses[-1]:.6f}",
                                      "elapsed time": round(time.time() - start2, 2),
                                      "total time": round(time.time() - start, 2)})
                    pbar.update(turbo)

                    if convergence_interval is not None:
                        active_losses = self.val_losses if validation else self.losses
                        if loop*turbo > 2 * convergence_interval:
                            # get means of last two intervals and standard deviation of last interval
                            average1 = jnp.mean(
                                active_losses[-convergence_interval:])
                            average2 = jnp.mean(
                                active_losses[-2 * convergence_interval: -convergence_interval])
                            std1 = jnp.std(
                                active_losses[-convergence_interval:])
                            # if the difference in averages is small compared to the statistical fluctuations,
                            # or the loss increases, stop training.
                            cond1 = jnp.abs(average2 - average1) <= std1 / \
                                jnp.sqrt(convergence_interval) / 2
                            cond2 = average1 > average2
                            if cond1 or cond2:
                                print(
                                    f'Training converged after {loop*turbo} steps')
                                converged = True
                                break

            if converged is False:
                print(f'Training has not converged after {n_iters} steps')
            params = carry[0]

        else:
            params_hist = []
            with tqdm(total=n_iters, desc="Training Progress",
                      postfix={"loss": 0.0, "elapsed time": 0.0, "total time": 0.0}) as pbar:
                for i in range(n_iters):
                    start2 = time.time()
                    key, subkey = jax.random.split(key, 2)
                    key_val, subkey_val = jax.random.split(key_val, 2)
                    params, state = self.opt.update(
                        params, state, **fixed_loss_kwargs, key=subkey)
                    v = self.loss(params, **fixed_val_kwargs,
                                  key=subkey_val) if validation else 0.

                    if self.optimizer == "GradientDescent":
                        key, subkey_gd = jax.random.split(key, 2)
                        l = self.loss(
                            params, **fixed_loss_kwargs, key=subkey_gd)
                    else:
                        l = state.value

                    # update parameter history
                    params_hist = params_hist + \
                        [params] if monitor_interval is not None and i % monitor_interval == 0 else params_hist

                    self.losses = jnp.concatenate(
                        (self.losses, jnp.array([l])))
                    self.val_losses = jnp.concatenate(
                        (self.val_losses, jnp.array([v])))
                    self.params_hist = params_hist

                    pbar.set_postfix({"loss": f"{l:.6f}", "elapsed time": round(time.time()-start2, 2),
                                     "total time": round(time.time()-start, 2)})
                    pbar.update(1)

                    if convergence_interval is not None:
                        active_losses = self.val_losses if validation else self.losses
                        if i > 2 * convergence_interval:
                            # get means of last two intervals and standard deviation of last interval
                            average1 = jnp.mean(
                                active_losses[-convergence_interval:])
                            average2 = jnp.mean(
                                active_losses[-2 * convergence_interval: -convergence_interval])
                            std1 = jnp.std(
                                active_losses[-convergence_interval:])
                            # if the difference in averages is small or loss increases, stop training.
                            cond1 = jnp.abs(average2 - average1) <= std1 / \
                                jnp.sqrt(convergence_interval) / 2
                            cond2 = average1 > average2
                            if cond1 or cond2:
                                print(f'Training converged after {i} steps')
                                converged = True
                                break

            if not converged:
                print(f'Training has not converged after {n_iters} steps')

        self.final_params = jnp.array(params)
        self.run_time = time.time()-start
