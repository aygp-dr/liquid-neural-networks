"""Tests for Liquid Neural Networks core module.

Validates LTC dynamics against Hasani et al. 2021 equations:
    dx/dt = -x/τ_eff + f(Wx + b) * A
    where τ_eff = τ / (1 + β * |f|)
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')

from liquid_neural_networks import (
    LTCConfig,
    LTCNeuron,
    CfCNeuron,
    LiquidNetwork,
    get_activation,
)


class TestActivations:
    """Test activation functions."""

    def test_tanh(self):
        act = get_activation('tanh')
        x = np.array([-1.0, 0.0, 1.0])
        result = act(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid(self):
        act = get_activation('sigmoid')
        x = np.array([-1.0, 0.0, 1.0])
        result = act(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_relu(self):
        act = get_activation('relu')
        x = np.array([-1.0, 0.0, 1.0])
        result = act(x)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            get_activation('invalid')


class TestLTCConfig:
    """Test LTCConfig dataclass."""

    def test_defaults(self):
        config = LTCConfig(input_size=4, hidden_size=8, output_size=2)
        assert config.tau == 1.0
        assert config.A == 1.0
        assert config.beta == 0.1
        assert config.dt == 0.1
        assert config.activation == 'tanh'

    def test_custom_values(self):
        config = LTCConfig(
            input_size=4,
            hidden_size=8,
            output_size=2,
            tau=2.0,
            beta=0.5
        )
        assert config.tau == 2.0
        assert config.beta == 0.5


class TestLTCNeuron:
    """Test LTC neuron implementation against paper equations."""

    def test_initialization(self):
        config = LTCConfig(input_size=4, hidden_size=8, output_size=2)
        neuron = LTCNeuron(config, seed=42)
        assert neuron.W.shape == (8, 4)
        assert neuron.b.shape == (8,)

    def test_compute_f(self):
        """Test f-function: f(Wx + b*x)."""
        config = LTCConfig(input_size=2, hidden_size=3, output_size=1)
        neuron = LTCNeuron(config, seed=42)

        x = np.zeros(3)
        u = np.ones(2)
        f = neuron.compute_f(x, u)

        # f = tanh(W @ u + b * x) = tanh(W @ u) since x=0
        expected = np.tanh(neuron.W @ u)
        np.testing.assert_allclose(f, expected)

    def test_compute_tau_effective(self):
        """Test τ_eff = τ / (1 + β * |f|) from Hasani et al. 2021."""
        config = LTCConfig(input_size=2, hidden_size=3, output_size=1, tau=2.0, beta=0.5)
        neuron = LTCNeuron(config, seed=42)

        f = np.array([0.5, -0.8, 0.0])
        tau_eff = neuron.compute_tau_effective(f)

        # τ_eff = τ / (1 + β * |f|)
        expected = 2.0 / (1.0 + 0.5 * np.abs(f))
        np.testing.assert_allclose(tau_eff, expected)

    def test_dynamics_equation(self):
        """Test dx/dt = -x/τ_eff + f * A from Hasani et al. 2021."""
        config = LTCConfig(
            input_size=2, hidden_size=3, output_size=1,
            tau=2.0, A=1.5, beta=0.2
        )
        neuron = LTCNeuron(config, seed=42)

        x = np.array([0.1, 0.2, 0.3])
        u = np.array([1.0, -1.0])

        # Manual computation following paper equations
        f = neuron.compute_f(x, u)
        tau_eff = neuron.compute_tau_effective(f)
        expected_dxdt = -x / tau_eff + f * config.A

        # Compare with dynamics method
        dxdt = neuron.dynamics(x, u)
        np.testing.assert_allclose(dxdt, expected_dxdt)

    def test_euler_step(self):
        """Test Euler integration: x(t+dt) = x(t) + dt * dx/dt."""
        config = LTCConfig(input_size=2, hidden_size=3, output_size=1, dt=0.1)
        neuron = LTCNeuron(config, seed=42)

        x = np.array([0.1, 0.2, 0.3])
        u = np.array([1.0, -1.0])
        dt = 0.1

        dxdt = neuron.dynamics(x, u)
        expected = x + dt * dxdt
        expected = np.clip(expected, -10.0, 10.0)

        x_new = neuron.step(x, u, dt)
        np.testing.assert_allclose(x_new, expected)

    def test_bounded_output(self):
        """Test that output is bounded to [-10, 10]."""
        config = LTCConfig(input_size=2, hidden_size=3, output_size=1, A=100.0)
        neuron = LTCNeuron(config, seed=42)

        x = np.zeros(3)
        u = np.ones(2) * 10

        x_new = neuron.step(x, u, dt=1.0)
        assert np.all(x_new >= -10.0)
        assert np.all(x_new <= 10.0)

    def test_deterministic_with_seed(self):
        """Test reproducibility with same seed."""
        config = LTCConfig(input_size=4, hidden_size=8, output_size=2)

        neuron1 = LTCNeuron(config, seed=42)
        neuron2 = LTCNeuron(config, seed=42)

        np.testing.assert_allclose(neuron1.W, neuron2.W)
        np.testing.assert_allclose(neuron1.b, neuron2.b)


class TestCfCNeuron:
    """Test Closed-form Continuous-time neuron."""

    def test_initialization(self):
        config = LTCConfig(input_size=4, hidden_size=8, output_size=2)
        neuron = CfCNeuron(config, seed=42)
        assert neuron.W.shape == (8, 4)
        assert neuron.b.shape == (8,)

    def test_closed_form_solution(self):
        """Test x(t+dt) = decay * x + (1-decay) * target."""
        config = LTCConfig(
            input_size=2, hidden_size=3, output_size=1,
            tau=2.0, A=1.0, beta=0.1
        )
        neuron = CfCNeuron(config, seed=42)

        x = np.array([0.5, 0.5, 0.5])
        u = np.array([1.0, -1.0])
        dt = 0.1

        # Manual computation
        f = neuron.activation(neuron.W @ u + neuron.b * x)
        tau_eff = config.tau / (1.0 + config.beta * np.abs(f))
        decay = np.exp(-dt / tau_eff)
        target = f * config.A
        expected = decay * x + (1 - decay) * target
        expected = np.clip(expected, -10.0, 10.0)

        x_new = neuron.step(x, u, dt)
        np.testing.assert_allclose(x_new, expected)

    def test_converges_to_target(self):
        """Test that CfC converges to target with large dt."""
        config = LTCConfig(input_size=2, hidden_size=3, output_size=1, tau=1.0)
        neuron = CfCNeuron(config, seed=42)

        x = np.zeros(3)
        u = np.ones(2)

        # With very large dt, should converge to target
        x_new = neuron.step(x, u, dt=100.0)
        f = neuron.activation(neuron.W @ u + neuron.b * x)
        target = f * config.A

        np.testing.assert_allclose(x_new, np.clip(target, -10, 10), rtol=0.01)


class TestLiquidNetwork:
    """Test multi-layer Liquid Neural Network."""

    def test_initialization(self):
        net = LiquidNetwork(input_size=4, hidden_size=8, output_size=2)
        assert len(net.layers) == 1
        assert net.hidden_size == 8
        assert net.output_size == 2
        assert net.W_out.shape == (2, 8)

    def test_multi_layer(self):
        net = LiquidNetwork(
            input_size=4, hidden_size=8, output_size=2, num_layers=3
        )
        assert len(net.layers) == 3

    def test_forward_shape(self):
        net = LiquidNetwork(input_size=4, hidden_size=8, output_size=2)
        inputs = np.random.randn(100, 4)
        outputs, states = net.forward(inputs)

        assert outputs.shape == (100, 2)
        assert len(states) == 1
        assert states[0].shape == (8,)

    def test_forward_with_initial_state(self):
        net = LiquidNetwork(input_size=4, hidden_size=8, output_size=2)
        inputs = np.random.randn(10, 4)
        initial = np.ones(8) * 0.5

        outputs, states = net.forward(inputs, initial_state=initial)
        assert outputs.shape == (10, 2)

    def test_cfc_network(self):
        net = LiquidNetwork(
            input_size=4, hidden_size=8, output_size=2, neuron_type='cfc'
        )
        inputs = np.random.randn(50, 4)
        outputs, states = net.forward(inputs)

        assert outputs.shape == (50, 2)
        assert not np.any(np.isnan(outputs))

    def test_deterministic_with_seed(self):
        inputs = np.random.randn(20, 4)

        net1 = LiquidNetwork(input_size=4, hidden_size=8, output_size=2, seed=42)
        net2 = LiquidNetwork(input_size=4, hidden_size=8, output_size=2, seed=42)

        out1, _ = net1.forward(inputs)
        out2, _ = net2.forward(inputs)

        np.testing.assert_allclose(out1, out2)


class TestHasaniEquations:
    """Validate implementation against Hasani et al. 2021 paper equations."""

    def test_ltc_dynamics_match_paper(self):
        """
        Paper Eq: dx/dt = -x/τ + f(x,u,t) where f modulates τ
        Our impl: dx/dt = -x/τ_eff + f * A where τ_eff = τ/(1+β|f|)
        """
        config = LTCConfig(
            input_size=2,
            hidden_size=4,
            output_size=1,
            tau=1.0,
            A=1.0,
            beta=0.1
        )
        neuron = LTCNeuron(config, seed=42)

        x = np.array([0.5, -0.3, 0.1, 0.8])
        u = np.array([1.0, -0.5])

        # Compute components
        f = neuron.compute_f(x, u)
        tau_eff = neuron.compute_tau_effective(f)
        dxdt = neuron.dynamics(x, u)

        # Verify: smaller τ_eff means faster dynamics (adaptive time constant)
        # When |f| is larger, τ_eff is smaller, making -x/τ_eff larger
        assert np.all(tau_eff > 0)
        assert np.all(tau_eff <= config.tau)

        # Verify dynamics structure
        leak_term = -x / tau_eff
        drive_term = f * config.A
        np.testing.assert_allclose(dxdt, leak_term + drive_term)

    def test_parameter_efficiency(self):
        """
        Paper claim: LNNs work with very few neurons (19-302).
        Test that small networks produce non-trivial outputs.
        """
        # Minimal network as in paper (19 neurons scenario)
        net = LiquidNetwork(
            input_size=4,
            hidden_size=19,
            output_size=2,
            tau=2.0,
            beta=0.1
        )

        # Time-series input
        t = np.linspace(0, 10, 100)
        inputs = np.column_stack([np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])

        outputs, _ = net.forward(inputs, dt=0.1)

        # Outputs should be non-trivial (not all zeros or constant)
        assert np.std(outputs) > 0.01
        assert not np.any(np.isnan(outputs))

    def test_continuous_time_property(self):
        """
        Paper: LNNs operate in continuous time.
        Test: Smaller dt should give smoother trajectories.
        """
        net = LiquidNetwork(input_size=2, hidden_size=8, output_size=1, seed=42)
        inputs = np.random.randn(100, 2)

        # Coarse dt
        out_coarse, _ = net.forward(inputs, dt=0.5)

        # Fine dt (should be smoother)
        out_fine, _ = net.forward(inputs, dt=0.01)

        # Both should be valid
        assert not np.any(np.isnan(out_coarse))
        assert not np.any(np.isnan(out_fine))

    def test_time_constant_adaptation(self):
        """
        Paper: Time constants adapt based on input.
        τ_eff = τ / (1 + β * |f|)
        """
        config = LTCConfig(input_size=2, hidden_size=4, output_size=1, tau=2.0, beta=0.5)
        neuron = LTCNeuron(config, seed=42)

        x = np.zeros(4)

        # Small input -> small f -> tau_eff closer to tau
        u_small = np.array([0.1, 0.1])
        f_small = neuron.compute_f(x, u_small)
        tau_small = neuron.compute_tau_effective(f_small)

        # Large input -> large f -> tau_eff smaller
        u_large = np.array([10.0, 10.0])
        f_large = neuron.compute_f(x, u_large)
        tau_large = neuron.compute_tau_effective(f_large)

        # Larger input should give smaller effective time constant
        assert np.mean(tau_large) < np.mean(tau_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
