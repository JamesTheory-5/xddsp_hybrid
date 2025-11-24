# xddsp_hybrid
```python
# ddsp_hybrid.py
# ============================================================
# DDSP-Hybrid: differentiable harmonic + noise + reverb model
# using autograd, exportable to XDSP runtime.
#
# Dependencies:
#   pip install autograd
#
# (c) 2025 James Theory / you
# ============================================================

import numpy as onp          # plain NumPy for non-differentiable/runtime work
import autograd.numpy as np  # autograd-friendly NumPy
from autograd import grad


# ------------------------------------------------------------
# Simple harmonic oscillator bank (autograd-friendly)
# ------------------------------------------------------------

def oscillator(f, amp, t, phase=0.0):
    """
    f: (N,) array of instantaneous frequency in Hz
    amp: scalar amplitude
    t: (N,) time in seconds
    phase: scalar phase offset (radians)
    """
    return amp * np.sin(2.0 * np.pi * f * t + phase)


def harmonic_bank(f0, amplitudes, t, phases=None):
    """
    f0: (N,) fundamental frequency trajectory
    amplitudes: (K,) harmonic amplitudes (parameters to learn)
    t: (N,) time
    phases: optional (K,) phase offsets (radians)
    """
    K = amplitudes.shape[0]
    N = t.shape[0]

    if phases is None:
        phases = np.zeros(K)

    y = np.zeros(N)

    for k in range(1, K + 1):
        fk = f0 * k                # (N,)
        ak = amplitudes[k - 1]     # scalar
        phase_k = phases[k - 1]
        y = y + oscillator(fk, ak, t, phase_k)

    return y


# ------------------------------------------------------------
# DDSP-Hybrid class
# ------------------------------------------------------------

class DDSPHybrid:
    """
    DDSP-Hybrid:
      - Differentiable model built with autograd.numpy
      - Learns:
          * harmonic amplitudes (K,)
          * noise FIR kernel (M_noise,)
          * reverb IR kernel (L_reverb,)
      - Synth path:
          y_harm = harmonic_bank(f0, amps, t)
          y_noise = conv(white_noise, noise_fir)
          y_dry = y_harm + y_noise
          y = conv(y_dry, reverb_ir)
      - Trains on target signal with MSE.
      - Exports parameters in a dictionary suitable for
        passing into your XDSP runtime (harmonic/noise/reverb).
    """

    def __init__(self,
                 fs: float,
                 t: onp.ndarray,
                 f0: onp.ndarray,
                 target: onp.ndarray,
                 K_harm: int = 16,
                 M_noise: int = 64,
                 L_reverb: int = 256,
                 seed: int = 0):
        """
        fs      : sample rate (Hz)
        t       : (N,) time array in seconds
        f0      : (N,) fundamental frequency trajectory
        target  : (N,) target audio
        K_harm  : number of harmonics
        M_noise : FIR length for noise shaping
        L_reverb: length of reverb IR
        """

        self.fs = float(fs)
        self.t = np.array(t)
        self.f0 = np.array(f0)
        self.target = np.array(target)

        self.N = t.shape[0]
        self.K_harm = int(K_harm)
        self.M_noise = int(M_noise)
        self.L_reverb = int(L_reverb)

        # Fixed white-noise source (deterministic) for noise branch
        rng = onp.random.RandomState(seed)
        noise_src = rng.randn(self.N).astype(onp.float64)
        self.noise_source = np.array(noise_src)

        # Parameter vector: [harm_amps (K), noise_fir (M), reverb_ir (L)]
        theta0 = onp.zeros(self.K_harm + self.M_noise + self.L_reverb, dtype=onp.float64)
        # Some small random init helps:
        theta0[:self.K_harm] = 0.1 * rng.randn(self.K_harm)
        theta0[self.K_harm:self.K_harm + self.M_noise] = 0.01 * rng.randn(self.M_noise)
        theta0[self.K_harm + self.M_noise:] = 0.01 * rng.randn(self.L_reverb)

        self.theta = np.array(theta0)

    # ------------------ parameter utilities ------------------

    def unpack_theta(self, theta):
        """
        Split flat theta into:
          harmonic_amps (K,)
          noise_fir     (M_noise,)
          reverb_ir     (L_reverb,)
        """
        K = self.K_harm
        M = self.M_noise
        L = self.L_reverb

        harm = theta[:K]
        noise = theta[K:K + M]
        reverb = theta[K + M:K + M + L]
        return harm, noise, reverb

    # ------------------ synthesis graph ----------------------

    def synth_harmonic(self, theta):
        harmonic_amps, _, _ = self.unpack_theta(theta)
        # Optional positivity constraint:
        # amps = np.log1p(np.exp(harmonic_amps))  # softplus
        amps = harmonic_amps
        return harmonic_bank(self.f0, amps, self.t)

    def synth_noise(self, theta):
        _, noise_fir, _ = self.unpack_theta(theta)
        # Noise shaping: y_noise = conv(noise_src, noise_fir)
        # mode='same' keeps length N, autograd supports np.convolve
        y_noise = np.convolve(self.noise_source, noise_fir, mode="same")
        return y_noise

    def synth_reverb(self, dry, theta):
        _, _, reverb_ir = self.unpack_theta(theta)
        # Simple FIR reverb: y = conv(dry, reverb_ir)
        y = np.convolve(dry, reverb_ir, mode="same")
        return y

    def synth(self, theta):
        """
        Full differentiable synth:
          y = reverb(harmonic + noise)
        """
        y_harm = self.synth_harmonic(theta)
        y_noise = self.synth_noise(theta)
        y_dry = y_harm + y_noise
        y = self.synth_reverb(y_dry, theta)
        return y

    # ------------------ loss & training ----------------------

    def loss(self, theta):
        """
        MSE between synthesized signal and target.
        """
        y = self.synth(theta)
        return np.mean((y - self.target) ** 2)

    def train(self, n_iters: int = 200, lr: float = 1e-2, verbose: bool = True):
        """
        Simple gradient descent on full parameter vector theta.
        """
        def loss_wrapped(theta_flat):
            return self.loss(theta_flat)

        grad_loss = grad(loss_wrapped)

        theta = self.theta
        for it in range(n_iters):
            g = grad_loss(theta)
            L = loss_wrapped(theta)

            if verbose and (it % 5 == 0 or it == n_iters - 1):
                print(f"[DDSPHybrid] iter {it:4d}  loss = {L:.6e}")

            theta = theta - lr * g

        self.theta = theta

    # ------------------ export to XDSP runtime ----------------

    def export_to_xdsp(self):
        """
        Export learned parameters in a runtime-friendly dict.

        This is intentionally generic; you can plug these into
        your XDSP core however you like, for example:
          - harmonic_amps → your sine_table-based harmonic synth
          - noise_fir     → a FIR filter node for noise
          - reverb_ir     → a convolution / delay-network reverb
        """
        theta_np = onp.array(self.theta)
        harm, noise_fir, reverb_ir = self.unpack_theta(theta_np)

        export = {
            "fs": self.fs,
            "N": self.N,
            "K_harm": self.K_harm,
            "M_noise": self.M_noise,
            "L_reverb": self.L_reverb,
            "harmonic_amps": onp.array(harm, dtype=onp.float64),
            "noise_fir": onp.array(noise_fir, dtype=onp.float64),
            "reverb_ir": onp.array(reverb_ir, dtype=onp.float64),
        }
        return export

    # Convenience: synth with plain NumPy (runtime side)
    def synth_runtime(self):
        """
        Use the *learned* theta but return an onp.ndarray
        for easy use on the XDSP / NumPy side.
        """
        y = self.synth(self.theta)
        return onp.array(y, dtype=onp.float64)


# ------------------------------------------------------------
# Demo / sanity check
# ------------------------------------------------------------

def main():
    fs = 48000.0
    duration = 0.1  # seconds
    N = int(fs * duration)
    t = onp.linspace(0.0, duration, N, endpoint=False)

    # Fundamental trajectory: constant 440 Hz
    f0 = onp.ones(N, dtype=onp.float64) * 440.0

    # Target: a slightly bright 440 Hz tone through a small "room"
    base = onp.sin(2.0 * onp.pi * 440.0 * t)
    bright = 0.3 * onp.sin(2.0 * onp.pi * 880.0 * t)
    target_dry = base + bright

    # Simple "room" IR (for building the target) - non-differentiable side
    L_rev_true = 128
    rng = onp.random.RandomState(123)
    ir_true = onp.exp(-onp.linspace(0, 1.5, L_rev_true)) * (0.2 + 0.8 * rng.rand(L_rev_true))
    target = onp.convolve(target_dry, ir_true, mode="same")

    # Create DDSPHybrid model
    model = DDSPHybrid(
        fs=fs,
        t=t,
        f0=f0,
        target=target,
        K_harm=8,
        M_noise=32,
        L_reverb=64,
        seed=42,
    )

    print("Training DDSPHybrid (harmonics + noise + reverb)...")
    model.train(n_iters=100, lr=5e-2, verbose=True)

    # Synthesize with learned parameters
    y_hat = model.synth_runtime()
    final_loss = onp.mean((y_hat - target) ** 2)
    print("\nFinal training loss:", final_loss)

    # Export to XDSP runtime
    params = model.export_to_xdsp()
    print("\nExported parameters for XDSP runtime:")
    print("  fs:", params["fs"])
    print("  K_harm:", params["K_harm"])
    print("  harmonic_amps:", params["harmonic_amps"])
    print("  noise_fir length:", len(params["noise_fir"]))
    print("  reverb_ir length:", len(params["reverb_ir"]))

    # At this point, you can:
    #   - Use params["harmonic_amps"] to build a harmonic patch
    #     with your sine_table_* functions in xdsp_core.py
    #   - Use params["noise_fir"] as an FIR for a noise channel
    #   - Use params["reverb_ir"] for a convolving reverb or
    #     to fit a delay-network in your XDSP runtime.


if __name__ == "__main__":
    main()

```
