import jax
import jax.numpy as jnp
import pandas as pd
from iqpopt.gen_qml.utils import median_heuristic
from iqpopt.utils import *
import matplotlib.pyplot as plt
import numpy as np
import iqpopt.iqp_optimizer as iqp
import iqpopt.training as train
import iqpopt.gen_qml as gen
from itertools import product
import os

if not os.path.exists('./plots'):
    os.makedirs('./plots')

#### A small training to get some decent params

dataset_path = f"../paper/datasets/ising/spin_glass_dataset_small/ising_spin_glass_N_64_T_0.7_train.csv"
X_train = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
X_train = jnp.array((1 + X_train) // 2, dtype=int)

# 16 qubit model
n_qubits = X_train.shape[-1]
sigma = median_heuristic(np.array(X_train[:1000]))
gates = gates_from_covariance(X_train, 500)

circuit = iqp.IqpSimulator(n_qubits, gates, spin_sym=False)
optimizer = "Adam"
stepsize = 0.01
n_iters = 500
params_init = jnp.array(np.random.normal(0, 1/len(gates), len(gates)))

loss_kwargs = {
    "params": params_init,
    "iqp_circuit": circuit,
    "ground_truth": X_train,
    "sigma": sigma,
    "n_ops": 1000,
    "n_samples": 1000
}

trainer = train.Trainer(optimizer, gen.mmd_loss_iqp, stepsize, opt_jit=False)
trainer.train(n_iters, loss_kwargs, val_kwargs=None,
              monitor_interval=None, turbo=None,
              convergence_interval=200)

params = jnp.array(trainer.final_params)


##### histograms for all the different options

settings_names = ['spin_smy', 'sparse']

for settings in product([True, False], repeat=len(settings_names)):
    circuit = iqp.IqpSimulator(n_qubits, gates, spin_sym=settings[0],
                                                sparse=settings[1], bitflip=True)

    height = 5 #height of the bar that marks the mean and stds on the plot
    trials = 100
    n_ground_truth_samples = 100 #the number of data points to sub sample in the estimates
    n_circuit_samples = 100 #the number of samples from the iqp circiut
    n_ops = 1000 #for the iqp estimation
    n_samples = 10  #for the iqp estimation

    sample_mmds = [] #
    iqp_mmds = []

    for t in range(trials):
        samples = circuit.sample(params, n_circuit_samples)
        idxs = np.random.choice(np.arange(X_train.shape[0]),n_ground_truth_samples)
        sample_mmds.append(gen.mmd_loss_samples(samples,X_train[idxs], sigma))
        iqp_mmds.append(gen.mmd_loss_iqp(jnp.array(params), circuit, X_train[idxs], sigma, n_ops, n_samples,
                                      key=jax.random.PRNGKey(np.random.randint(999999))))


    plt.clf()
    plt.hist(iqp_mmds, bins=20, alpha=0.5, label='iqp')
    plt.hist(sample_mmds, bins=20, alpha=0.5, label='sample')
    plt.legend()
    plt.title('spin_sym: '+str(settings[0]) + ', sparse: '+str(settings[1]) )

    plt.vlines([np.mean(iqp_mmds), np.mean(iqp_mmds) + np.std(iqp_mmds), np.mean(iqp_mmds) - np.std(iqp_mmds)], 0,
               height, colors='blue')
    plt.vlines([np.mean(sample_mmds), np.mean(sample_mmds) + np.std(sample_mmds),
                np.mean(sample_mmds) - np.std(sample_mmds)], 0, height, colors='orange')

    print(' ')
    print('spin_sym: '+str(settings[0]) + ' sparse: '+str(settings[1]) )

    print('mean (iqp mmd): ' + str(np.mean(iqp_mmds)) + ' +/- ' + str(np.std(iqp_mmds)))
    print('mean (sample mmd): ' + str(np.mean(sample_mmds)) + ' +/- ' + str(np.std(sample_mmds)))

    plt.savefig('./plots/mmd_test_spin_sym_'+str(settings[0]) +'_sparse_'+str(settings[1]) +'.png')






