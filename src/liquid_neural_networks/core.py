"""Core module for Liquid Neural Networks.

Pure NumPy/SciPy implementation of Liquid Time-Constant (LTC) neurons
based on Hasani et al. "Liquid Time-constant Networks" (2021).

The core dynamics follow:
    dx/dt = -x/τ_eff + f(Wx + b) * A
    where τ_eff = τ / (1 + β * |f|)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class LTCConfig:
    """Configuration for LTC neurons.

    Attributes:
        input_size: Dimension of input vectors
        hidden_size: Number of hidden neurons
        output_size: Dimension of output vectors
        tau: Base time constant (default: 1.0)
        A: Amplitude scaling factor (default: 1.0)
        beta: Time constant adaptation rate (default: 0.1)
        dt: Integration time step (default: 0.1)
        activation: Activation function name ('tanh', 'sigmoid', 'relu')
    """
    input_size: int
    hidden_size: int
    output_size: int
    tau: float = 1.0
    A: float = 1.0
    beta: float = 0.1
    dt: float = 0.1
    activation: str = 'tanh'


def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Get activation function by name."""
    activations = {
        'tanh': np.tanh,
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


class LTCNeuron:
    """Liquid Time-Constant Neuron.

    Implements the LTC dynamics from Hasani et al. 2021:
        dx/dt = -x/τ_eff + f(Wx + b) * A
        where τ_eff = τ / (1 + β * |f|)

    Example:
        >>> config = LTCConfig(input_size=4, hidden_size=8, output_size=2)
        >>> neuron = LTCNeuron(config)
        >>> x = np.zeros(8)  # hidden state
        >>> u = np.random.randn(4)  # input
        >>> x_new = neuron.step(x, u, dt=0.1)
    """

    def __init__(self, config: LTCConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (config.input_size + config.hidden_size))
        self.W = self.rng.normal(0, scale, (config.hidden_size, config.input_size))
        self.b = self.rng.normal(0, 0.01, (config.hidden_size,))

        self.activation = get_activation(config.activation)

    def compute_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute the f-function: f(Wx + b*x)."""
        return self.activation(self.W @ u + self.b * x)

    def compute_tau_effective(self, f: np.ndarray) -> np.ndarray:
        """Compute effective time constant: τ_eff = τ / (1 + β * |f|)."""
        return self.config.tau / (1.0 + self.config.beta * np.abs(f))

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute dx/dt = -x/τ_eff + f * A."""
        f = self.compute_f(x, u)
        tau_eff = self.compute_tau_effective(f)
        return -x / tau_eff + f * self.config.A

    def step(self, x: np.ndarray, u: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Single Euler integration step.

        Args:
            x: Current hidden state (hidden_size,)
            u: Input vector (input_size,)
            dt: Time step (uses config.dt if not provided)

        Returns:
            New hidden state after one time step
        """
        dt = dt or self.config.dt
        dx = self.dynamics(x, u)
        x_new = x + dt * dx
        return np.clip(x_new, -10.0, 10.0)  # Bounded for stability


class CfCNeuron:
    """Closed-form Continuous-time Neuron.

    Uses closed-form solution instead of numerical integration:
        x(t+dt) = decay * x(t) + (1 - decay) * target
        where decay = exp(-dt/τ_eff)
    """

    def __init__(self, config: LTCConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / (config.input_size + config.hidden_size))
        self.W = self.rng.normal(0, scale, (config.hidden_size, config.input_size))
        self.b = self.rng.normal(0, 0.01, (config.hidden_size,))

        self.activation = get_activation(config.activation)

    def step(self, x: np.ndarray, u: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Closed-form integration step."""
        dt = dt or self.config.dt

        f = self.activation(self.W @ u + self.b * x)
        tau_eff = self.config.tau / (1.0 + self.config.beta * np.abs(f))

        decay = np.exp(-dt / tau_eff)
        target = f * self.config.A

        x_new = decay * x + (1 - decay) * target
        return np.clip(x_new, -10.0, 10.0)


class LiquidNetwork:
    """Multi-layer Liquid Neural Network.

    Example:
        >>> net = LiquidNetwork(input_size=10, hidden_size=32, output_size=2)
        >>> inputs = np.random.randn(100, 10)  # 100 time steps
        >>> outputs, states = net.forward(inputs)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        neuron_type: str = 'ltc',
        seed: Optional[int] = None,
        **kwargs
    ):
        self.rng = np.random.default_rng(seed)

        self.layers: List = []
        NeuronClass = LTCNeuron if neuron_type == 'ltc' else CfCNeuron

        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            config = LTCConfig(
                input_size=layer_input,
                hidden_size=hidden_size,
                output_size=hidden_size,
                **kwargs
            )
            self.layers.append(NeuronClass(config, seed=self.rng.integers(0, 2**31)))

        # Output projection
        self.W_out = self.rng.normal(0, 0.1, (output_size, hidden_size))
        self.b_out = np.zeros(output_size)

        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Args:
            inputs: Input sequence (seq_len, input_size)
            initial_state: Initial hidden state (optional)
            dt: Time step (optional)

        Returns:
            outputs: Output sequence (seq_len, output_size)
            final_states: Final hidden states for each layer
        """
        seq_len = inputs.shape[0]

        # Initialize hidden states
        states = [np.zeros(self.hidden_size) for _ in self.layers]
        if initial_state is not None:
            states[0] = initial_state

        outputs = []

        for t in range(seq_len):
            u = inputs[t]

            for i, layer in enumerate(self.layers):
                layer_input = u if i == 0 else states[i-1]
                states[i] = layer.step(states[i], layer_input, dt)

            # Output projection
            y = self.W_out @ states[-1] + self.b_out
            outputs.append(y)

        return np.array(outputs), states


def greet(name: str = "World") -> str:
    """Return a greeting message."""
    return f"Hello, {name}! Welcome to Liquid Neural Networks."


def main() -> None:
    """Demo of LTC network on synthetic time-series."""
    print("Liquid Neural Networks - NumPy Implementation")
    print("=" * 50)

    # Create network
    net = LiquidNetwork(
        input_size=4,
        hidden_size=8,
        output_size=1,
        num_layers=1,
        tau=2.0,
        beta=0.1
    )

    # Generate synthetic sinusoidal input
    t = np.linspace(0, 10, 100)
    inputs = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t)
    ])

    # Forward pass
    outputs, final_states = net.forward(inputs, dt=0.1)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final state norm: {np.linalg.norm(final_states[0]):.4f}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print("\nNetwork successfully processed time-series data!")


if __name__ == "__main__":
    main()
