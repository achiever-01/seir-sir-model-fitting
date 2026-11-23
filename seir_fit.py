"""
SEIR vs SIR: simulate, fit by maximum likelihood, compare models, profile R0.

Workflow
1. Simulate "truth" with a deterministic SEIR model and add negative-binomial
   observation noise to weekly case counts.
2. Fit SEIR and SIR by maximum likelihood (Nelder-Mead, multiple starts).
3. Check parameter recovery, compare models with AIC.
4. Profile likelihood for R0 (approximate 95% interval).

Time unit: days. Observations: weekly counts.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import nbinom
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 50_000            # population size (assumed known)
SIGMA = 1 / 3         # 1 / latent period (days), assumed known
T_END = 154           # 22 weeks
WEEK_T = np.arange(0, T_END + 1, 7)   # weekly boundaries (23 points -> 22 counts)

# ----------------------------------------------------------------- models
def rhs_seir(t, y, beta, gamma):
    S, E, I, R, C = y
    inf = beta * S * I / N
    return [-inf, inf - SIGMA * E, SIGMA * E - gamma * I, gamma * I, SIGMA * E]

def rhs_sir(t, y, beta, gamma):
    S, I, R, C = y
    inf = beta * S * I / N
    return [-inf, inf - gamma * I, gamma * I, inf]

def weekly_incidence(model, R0, gamma, I0):
    """Expected weekly new cases (SEIR: symptom onsets; SIR: new infections)."""
    beta = R0 * gamma
    if model == "SEIR":
        y0 = [N - 2 * I0, I0, I0, 0.0, 0.0]
        sol = solve_ivp(rhs_seir, (0, T_END), y0, t_eval=WEEK_T,
                        args=(beta, gamma), rtol=1e-8, atol=1e-8)
        C = sol.y[4]
    else:
        y0 = [N - I0, I0, 0.0, 0.0]
        sol = solve_ivp(rhs_sir, (0, T_END), y0, t_eval=WEEK_T,
                        args=(beta, gamma), rtol=1e-8, atol=1e-8)
        C = sol.y[3]
    return np.maximum(np.diff(C), 1e-9)

# ------------------------------------------------------------ likelihood
def neg_loglik(theta, model, data, fixed=None):
    """theta = log(R0), log(gamma), log(I0), log(k). `fixed` pins log(R0)."""
    th = np.array(theta, dtype=float)
    if fixed is not None:
        th = np.insert(th, 0, fixed)
    R0, gamma, I0, k = np.exp(th)
    if not (0.2 < R0 < 20 and 1 / 30 < gamma < 2 and 0.1 < I0 < N / 10 and 0.1 < k < 1e4):
        return 1e12
    mu = weekly_incidence(model, R0, gamma, I0)
    p = k / (k + mu)
    return -np.sum(nbinom.logpmf(data, k, p))

def fit(model, data, n_starts=8, fixed=None):
    best = None
    for _ in range(n_starts):
        start = [np.log(rng.uniform(0.15, 0.5)), np.log(rng.uniform(1, 100)), np.log(rng.uniform(5, 50))]
        if fixed is None:
            start = [np.log(rng.uniform(1.2, 4))] + start
        res = minimize(neg_loglik, start, args=(model, data, fixed),
                       method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return best

# ---------------------------------------------------------- 1. simulate
TRUE = dict(R0=2.0, gamma=1 / 4, I0=10.0, k=30.0)
mu_true = weekly_incidence("SEIR", TRUE["R0"], TRUE["gamma"], TRUE["I0"])
data = nbinom.rvs(TRUE["k"], TRUE["k"] / (TRUE["k"] + mu_true), random_state=rng)

# ---------------------------------------------------------- 2. fit both
fits = {m: fit(m, data) for m in ("SEIR", "SIR")}
for m, res in fits.items():
    R0, gamma, I0, k = np.exp(res.x)
    aic = 2 * res.fun + 2 * len(res.x)
    print(f"{m}: R0={R0:.2f} gamma={gamma:.3f} I0={I0:.1f} k={k:.1f} "
          f"negLL={res.fun:.2f} AIC={aic:.2f}")
print(f"truth: R0={TRUE['R0']} gamma={TRUE['gamma']} I0={TRUE['I0']} k={TRUE['k']}")

# ------------------------------------------- 3. profile likelihood for R0
grid = np.linspace(1.5, 3.0, 25)
prof = []
for r in grid:
    res = fit("SEIR", data, n_starts=3, fixed=np.log(r))
    prof.append(res.fun)
prof = np.array(prof)
ok = grid[prof - prof.min() < 1.92]
print(f"SEIR R0 approx. 95% profile-likelihood interval: {ok.min():.2f} to {ok.max():.2f}")

# --------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
wk = np.arange(1, len(data) + 1)
ax[0].scatter(wk, data, color="k", label="Simulated data")
for m, c in (("SEIR", "tab:blue"), ("SIR", "tab:orange")):
    R0, gamma, I0, k = np.exp(fits[m].x)
    ax[0].plot(wk, weekly_incidence(m, R0, gamma, I0), color=c, label=f"{m} fit")
ax[0].set_xlabel("Week"); ax[0].set_ylabel("Weekly cases"); ax[0].legend()
ax[0].set_title("Model fits")
ax[1].plot(grid, prof - prof.min())
ax[1].axhline(1.92, ls="--", color="grey")
ax[1].axvline(TRUE["R0"], ls=":", color="red", label="True R0")
ax[1].set_xlabel("R0"); ax[1].set_ylabel("Profile negLL - min"); ax[1].legend()
ax[1].set_title("SEIR profile likelihood for R0")
plt.tight_layout()
plt.savefig("seir_fit.png", dpi=150)
