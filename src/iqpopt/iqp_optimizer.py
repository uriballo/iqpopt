from jax._src.typing import Array
from scipy.sparse import csr_matrix, dok_matrix
import pennylane as qml
import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from . import utils


class IqpSimulator:
    """ Class that creates an IqpSimulator object corresponding to a parameterized IQP circuit"""

    def __init__(self, n_qubits: int, gates: list, device: str = "lightning.qubit",
                 spin_sym: bool = False, init_gates: list = None, init_coefs: list = None,
                 sparse: bool = False, bitflip: bool = False, encoding: bool = False,
                 encoding_repeats: int = 2):
        """
        Args:
            n_qubits (int): Total number of qubits of the circuit.
            gates (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            device (str, optional): Pennylane device used for calculating probabilities and sampling.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                1/sqrt(2)(|00...0> + |11...1>) is used in place of |00...0>.
            init_gates (list[list[list[int]]], optional): A specification of gates of the same form as the gates argument. The
                parameters of these gates are kept fixed according to init_coefs.
            init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
            sparse (bool, optional): If True, generators and ops are always stored in sparse matrix format, leading
                to better memory efficiency and potentially faster runtime.
            bitflip (bool, optional): If True, the circuit is equivalent to a classical stochastic model where the
                gates correspond to correlated bitflips.
            encoding (bool, optional): If True, uses the IQP encoding template at the beginning of the circuit.
            encoding_repeats (int, optional): Number of repetitions of the IQP encoding template.

        Raises:
            Exception: when gates and params have a different number of elements.
            Exception: when init_gates and init_coefs have a different number of elements.
        """
        self.n_qubits = n_qubits
        self.gates = gates
        self.n_gates = len(gates)
        self.sparse = sparse
        self.init_gates = init_gates
        self.init_coefs = jnp.array(init_coefs) if init_coefs is not None else None
        self.device = device
        self.spin_sym = spin_sym
        self.bitflip = bitflip

        self.generators = []
        self.generators_sp = None

        len_gen_init = 0
        if self.init_gates is not None:

            if len(self.init_coefs) != len(self.init_gates):
                raise ValueError("init_gates and init_coefs must both have the same number of elements")

            len_gen_init = sum(1 for gate in self.init_gates for _ in gate)

        len_gen = sum(1 for gate in gates for _ in gate) + len_gen_init
        self.par_transform = False if max([len(gate) for gate in self.gates]) == 1 and self.init_gates is None else True

        if sparse:
            generators_dok = dok_matrix((len_gen, n_qubits), dtype='float64')
            i = 0
            for gate in gates:
                for gen in gate:
                    for j in gen:
                        generators_dok[i, j] = 1
                    i += 1

            if self.init_gates is not None:
                for gate in self.init_gates:
                    for gen in gate:
                        for j in gen:
                            generators_dok[i, j] = 1
                        i += 1

            # convert to csr format
            self.generators_sp = generators_dok.tocsr()

        else:
            # Transformation of the input gates to generators
            # convert the gates to a list of arrays
            self.gates_as_arrays = utils.gate_lists_to_arrays(gates, n_qubits)

            # store all generators
            self.generators = []
            for gens in self.gates_as_arrays:
                for gen in gens:
                    self.generators.append(gen)

            if self.init_gates is not None:
                self.init_gates_as_arrays = utils.gate_lists_to_arrays(self.init_gates, n_qubits)

                # could this be more efficient? we are potentially storing the same generators more than once
                for gens in self.init_gates_as_arrays:
                    for gen in gens:
                        self.generators.append(gen)

            self.generators = jnp.array(self.generators)

        if self.par_transform:
            # Transformation matrix from the number of independent parameters to the number of total generators
            self.trans_par = np.zeros((len_gen, len(gates)))
            i = 0
            for j, gens in enumerate(gates):
                for gen in gens:
                    # Matrix that linearly transforms the vector of parameters that are trained into the vector of parameters that apply to the generators
                    self.trans_par[i, j] = 1
                    i += 1
            self.trans_par = jnp.array(self.trans_par)

        if self.init_gates is not None:
            # Matrix that transforms the static parameters (initial coefficients) into a vector of size generators so it can be summed with the variational parameters
            trans_coef = np.zeros((len_gen, len(self.init_gates)))
            i = len(self.generators) - len_gen_init
            for j, gens in enumerate(self.init_gates):
                for gen in gens:
                    trans_coef[i, j] = 1
                    i += 1
            trans_coef = jnp.array(trans_coef)

            self.init_trans_coef = trans_coef @ self.init_coefs
            
        self.encoding = encoding
        self.encoding_repeats = encoding_repeats

    def iqp_circuit(self, params: jnp.ndarray, features: jnp.ndarray = None):
        """IQP circuit in pennylane form.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            features (jnp.ndarray, optional): The features to be encoded in the circuit. Defaults to None.
        """
        # Does it matter if its before or after the Hadamards?
        if self.encoding:
            if features is None:
                raise ValueError("features must be provided for encoding circuits")
            if len(features) != self.n_qubits:
                raise ValueError("features must have the same length as n_qubits")
            qml.IQPEmbedding(features, wires=range(self.n_qubits), n_repeats=self.encoding_repeats)
            
        if self.spin_sym:
            qml.PauliRot(2*jnp.pi/4, "Y"+"X"*(self.n_qubits-1), wires=range(self.n_qubits))

        for i in range(self.n_qubits):
            qml.Hadamard(i)

        if self.init_gates is not None:
            for par, gate in zip(self.init_coefs, self.init_gates):
                for gen in gate:
                    qml.MultiRZ(2*par, wires=gen)

        for par, gate in zip(params, self.gates):
            for gen in gate:
                qml.MultiRZ(2*par, wires=gen)

        for i in range(self.n_qubits):
            qml.Hadamard(i)

    def sample(self, params: jnp.ndarray, features: jnp.ndarray = None, shots: int = 1) -> jnp.ndarray:
        """Sample the IQP circuit using state vector simulation in Pennylane.
        Only possible for circuits with small numbers of qubits.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            features (jnp.ndarray, optional): The features to be encoded in the circuit. Defaults to None
            shots (int): Number of samples that are output. Defaults to 1.

        Returns:
            jnp.ndarray: The bitstring samples.
        """
        if self.bitflip:
            samples = []
            for _ in range(shots):
                sample = np.zeros(self.n_qubits)
                if self.spin_sym:
                    sample = (sample+1) % 2 if np.random.rand() > 0.5 else sample

                if self.init_gates is not None:
                    for par, gate in zip(self.init_coefs, self.init_gates):
                        for gen in gate:
                            if np.random.rand() < jnp.sin(par) ** 2:
                                sample[gen] = sample[gen]+1
                                sample = sample % 2

                for par, gate in zip(params, self.gates):
                    for gen in gate:
                        if np.random.rand() < jnp.sin(par) ** 2:
                            sample[gen] = sample[gen] + 1
                            sample = sample % 2
                samples.append(sample)
            return jnp.array(samples)

        else:
            dev = qml.device(self.device, wires=self.n_qubits, shots=shots)

            @qml.qnode(dev)
            def sample_circuit(params):
                self.iqp_circuit(params, features)
                return qml.sample(wires=range(self.n_qubits))
            return sample_circuit(params)

    def probs(self, params: jnp.ndarray, features: jnp.ndarray = None) -> jnp.ndarray:
        """Returns the probabilities of all possible bitstrings using state vector simulation in
        PennyLane. Only possible for circuits with small numbers of qubits.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            features (jnp.ndarray, optional): The features to be encoded in the circuit. Defaults to None.

        Returns:
            jnp.ndarray: Probabilities of all possible bitstrings.
        """

        if self.bitflip:
            raise NotImplementedError(
                "probs not implemented for bitflip circuits")
        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev)
        def probs_circuit(params):
            self.iqp_circuit(params, features)
            return qml.probs(wires=range(self.n_qubits))
        return probs_circuit(params)

    def __op_expval_indep(self, params: jnp.ndarray, ops: jnp.ndarray,
                          n_samples: int, key: Array, return_samples) -> list:
        """
        Batch evaluate an array of ops in the same way as self.op_expval_batch, but using independent randomness
        for each estimator. The estimators for each op are therefore uncorrelated.
        """

        def update(carry, op):
            key1, key2 = jax.random.split(carry, 2)
            expval = self.op_expval_batch(
                params, op, n_samples, key1, False, return_samples)
            return key2, expval

        if self.sparse:
            expvals = []
            stds = []
            for op in ops:
                key, val = update(key, op)
                if return_samples:
                    expvals.append(val[0])
                else:
                    expvals.append(val[0][0])
                    stds.append(val[1][0])

            if return_samples:
                return jnp.array(expvals)
            else:
                return jnp.array(expvals), jnp.array(stds)/jnp.sqrt(n_samples)

        else:
            _, op_expvals = jax.lax.scan(update, key, ops)

            if return_samples:
                return op_expvals
            else:
                return op_expvals[0], op_expvals[1]

    def op_expval_batch(self, params: jnp.ndarray, ops: jnp.ndarray, n_samples: int,
                        key: Array, indep_estimates: bool = False,
                        return_samples: bool = False) -> list:
        """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
        by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
        operator acts.
        The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
        with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
        uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.

        Args:
            params (jnp.ndarray): The parameters of the IQP gates.
            ops (jnp.ndarray): Operator/s for those we want to know the expected value.
            n_samples (int): Number of samples used to calculate the IQP expectation value.
            key (Array): Jax key to control the randomness of the process.
            indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.

        Returns:
            list: List of Vectors. The expected value of each op and its standard deviation.
        """

        if indep_estimates and not self.bitflip:
            return self.__op_expval_indep(params, ops, n_samples, key, return_samples)

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        effective_params = self.trans_par @ params if self.par_transform else params
        effective_params = effective_params + self.init_trans_coef if self.init_gates is not None else effective_params

        if self.bitflip:

            if self.sparse or isinstance(ops, csr_matrix):

                if isinstance(ops, csr_matrix):
                    if self.generators_sp is None:
                        self.generators_sp = csr_matrix(self.generators)

                else:
                    ops = csr_matrix(ops)

                ops_gen = ops.dot(self.generators_sp.T)
                
                if self.spin_sym:
                    ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
                    
                del ops

                ops_gen.data %= 2
                ops_gen = ops_gen.toarray()

            else:
                ops_gen = (ops @ self.generators.T) % 2
                if self.spin_sym:
                    ops_sum = jnp.sum(ops, axis=-1)
                
            par_ops_gates = 2 * effective_params * ops_gen

            expvals = jnp.prod(jnp.cos(par_ops_gates), axis=-1)

            if self.spin_sym:
                # flip expvals of odd operators with prob 1/2
                odd_ops = 1 - 2 * (ops_sum % 2)
                expvals = 0.5*expvals + 0.5*odd_ops*expvals

            if return_samples:
                return jnp.expand_dims(expvals, -1)
            else:
                return expvals, jnp.zeros(ops.shape[0])


        elif self.sparse or isinstance(ops, csr_matrix):

            if isinstance(ops, csr_matrix):
                samples = csr_matrix(samples)
                if self.generators_sp is None:
                    self.generators_sp = csr_matrix(self.generators)
            else:
                ops = csr_matrix(ops)
                samples = csr_matrix(samples)

            ops_gen = ops.dot(self.generators_sp.T)
            ops_gen.data %= 2
            ops_gen = ops_gen.toarray()

            samples_gates = samples.dot(self.generators_sp.T)
            samples_gates.data = 2 * (samples_gates.data % 2)
            samples_gates = samples_gates.toarray()
            samples_gates = 1 - samples_gates

            if self.spin_sym:
                ops_sum = np.squeeze(np.asarray(ops.sum(axis=-1)))
                samples_sum = np.squeeze(np.asarray(samples.sum(axis=-1)))
                samples_len = samples.shape[0]
            del ops            
            del samples

        else:
            ops_gen = (ops @ self.generators.T) % 2
            samples_gates = 1 - 2 * ((samples @ self.generators.T) % 2)
            if self.spin_sym:
                ops_sum = ops.sum(axis=-1)
                samples_sum = samples.sum(axis=-1)
                samples_len = samples.shape[0]

        if self.spin_sym:
            try:
                shape = (len(ops_sum), samples_len)
            except:
                shape = (samples_len, )
            
            ini_spin_sym = 2 - jnp.repeat(ops_sum, samples_len).reshape(shape) % 2 - 2*(samples_sum % 2)
            
        else:
            ini_spin_sym = 1
        # ini_spin_sym = jnp.where(self.spin_sym, ini_spin_sym, 1)

        par_ops_gates = 2 * effective_params * ops_gen
        expvals = ini_spin_sym * jnp.cos(par_ops_gates @ samples_gates.T)


        if return_samples:
            return expvals
        else:
            return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1)/jnp.sqrt(n_samples)

    def op_expval(self, params: jnp.ndarray, ops: jnp.ndarray, n_samples: int, key: Array,
                  indep_estimates: bool = False, return_samples: bool = False,
                  max_batch_ops: int = None, max_batch_samples: int = None) -> list:
        """Estimate the expectation values of a batch of Pauli-Z type operators. A set of l operators must be specified
        by an array of shape (l,n_qubits), where each row is a binary vector that specifies on which qubit a Pauli Z
        operator acts.
        The expectation values are estimated using a randomized method whose precision in controlled by n_samples,
        with larger values giving higher precision. Estimates are unbiased, however may be correlated. To request
        uncorrelated estimate, use indep_estimates=True at the cost of larger runtime.
        For large batches of operators or large values of n_samples, memory can be controlled by setting max_batch_ops
        and/or max_batch_samples to a fixed value.

        Args:
            params (jnp.ndarray): The parameters of the trainable gates of the circuit.
            ops (jnp.ndarray): Array specifying the operator/s for which to estimate the expectation values.
            n_samples (int): Number of samples used to calculate the IQP expectation values. Higher values result in
                higher precision.
            key (Array): Jax key to control the randomness of the process.
            indep_estimates (bool): Whether to use independent estimates of the ops in a batch.
            return_samples (bool): if True, an extended array that contains the values of the estimator for each
                of the n_samples samples is returned.
            max_batch_ops (int): Maximum number of operators in a batch. Defaults to None, which means taking all ops at once.
            max_batch_samples (int): Maximum number of samples in a batch. Defaults to None, which means taking all n_samples at once.

        Returns:
            list: List of Vectors. The expected value of each op and its standard deviation.
        """

        if max_batch_ops is None:
            max_batch_ops = len(ops)

        if max_batch_samples is None:
            max_batch_samples = n_samples

        if self.bitflip:
            n_samples = max_batch_samples

        if len(ops.shape) == 1:
            ops = ops.reshape(1, -1)

        if self.bitflip:
            expvals = jnp.empty((0, 1))
        else:
            expvals = jnp.empty((0, n_samples))

        for batch_ops in jnp.array_split(ops, np.ceil(ops.shape[0] / max_batch_ops)):
            tmp_expvals = jnp.empty((len(batch_ops), 0))
            for i in range(np.ceil(n_samples / max_batch_samples).astype(jnp.int64)):
                batch_n_samples = min(max_batch_samples, n_samples - i * max_batch_samples)
                key, subkey = jax.random.split(key, 2)
                batch_expval = self.op_expval_batch(
                    params, batch_ops, batch_n_samples, subkey,
                    indep_estimates, return_samples=True
                )
                tmp_expvals = jnp.concatenate((tmp_expvals, batch_expval), axis=-1)
            expvals = jnp.concatenate((expvals, tmp_expvals), axis=0)

        if self.bitflip:
            if return_samples:
                return expvals
            else:
                return jnp.mean(expvals, axis=-1), jnp.zeros(len(ops))
        else:
            if return_samples:
                return expvals
            else:
                return jnp.mean(expvals, axis=-1), jnp.std(expvals, axis=-1, ddof=1)/jnp.sqrt(n_samples)
