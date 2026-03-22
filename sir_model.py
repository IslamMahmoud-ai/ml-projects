"""
SIR Epidemic Model — Simulation & Machine Learning
Author: Islam Mahmoud
GitHub: IslamMahmoud-ai
"""

import numpy as np
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# ─── 1. SIR Differential Equations ───────────────────────────────────────────

def sir_equations(y, t, beta, gamma, N):
    """
    SIR model ordinary differential equations.
    S: Susceptible, I: Infected, R: Removed
    beta : transmission rate
    gamma: recovery rate
    N    : total population
    """
    S, I, R = y
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return dS, dI, dR


def simulate_sir(beta, gamma, N=10_000, I0=10, days=160):
    """Simulate one SIR epidemic and return a DataFrame."""
    S0 = N - I0
    R0_init = 0
    y0 = S0, I0, R0_init
    t = np.linspace(0, days, days)
    solution = odeint(sir_equations, y0, t, args=(beta, gamma, N))
    S, I, R = solution.T
    return pd.DataFrame({"t": t, "S": S, "I": I, "R": R,
                         "beta": beta, "gamma": gamma, "N": N})


# ─── 2. Generate Synthetic Dataset ───────────────────────────────────────────

def generate_dataset(n_samples=500, N=10_000, days=160, seed=42):
    """
    Generate synthetic epidemics at random (beta, gamma) parameter points.
    Returns feature matrix X and target matrix y.
    """
    rng = np.random.default_rng(seed)
    betas  = rng.uniform(0.1, 0.5, n_samples)
    gammas = rng.uniform(0.05, 0.3, n_samples)

    records = []
    for beta, gamma in zip(betas, gammas):
        df = simulate_sir(beta, gamma, N=N, days=days)
        peak_I   = df["I"].max()
        peak_day = df["I"].idxmax()
        final_R  = df["R"].iloc[-1]
        records.append({
            "beta": beta,
            "gamma": gamma,
            "R0": beta / gamma,          # basic reproduction number
            "peak_infected": peak_I,
            "peak_day": peak_day,
            "final_removed": final_R,
        })

    return pd.DataFrame(records)


# ─── 3. Train ML Model ────────────────────────────────────────────────────────

def train_model(df):
    """
    Train a Random Forest to predict epidemic outcomes
    from (beta, gamma, R0) parameters.
    """
    features = ["beta", "gamma", "R0"]
    targets  = ["peak_infected", "peak_day", "final_removed"]

    X = df[features].values
    results = {}

    for target in targets:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[target] = {"model": model, "rmse": rmse, "r2": r2,
                           "y_test": y_test, "y_pred": y_pred}
        print(f"  {target:20s} → RMSE: {rmse:10.2f}   R²: {r2:.4f}")

    return results


# ─── 4. Visualisation ─────────────────────────────────────────────────────────

def plot_sir_curve(beta=0.3, gamma=0.1, N=10_000, save_path=None):
    """Plot a single SIR epidemic curve."""
    df = simulate_sir(beta, gamma, N=N)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["t"], df["S"], label="Susceptible", color="#3B8BD4", lw=2)
    ax.plot(df["t"], df["I"], label="Infected",    color="#E24B4A", lw=2)
    ax.plot(df["t"], df["R"], label="Removed",     color="#1D9E75", lw=2)
    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title(f"SIR Epidemic Model  (β={beta}, γ={gamma}, R₀={beta/gamma:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predictions(results, save_path=None):
    """Plot actual vs predicted for all targets."""
    targets = list(results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(15, 5))
    colors = ["#3B8BD4", "#E24B4A", "#1D9E75"]

    for ax, target, color in zip(axes, targets, colors):
        y_test = results[target]["y_test"]
        y_pred = results[target]["y_pred"]
        r2     = results[target]["r2"]
        ax.scatter(y_test, y_pred, alpha=0.5, color=color, s=20)
        lims = [min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target}\nR² = {r2:.4f}")
        ax.grid(alpha=0.3)

    plt.suptitle("ML Model: Actual vs Predicted", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─── 5. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SIR Epidemic Model — ML Parameter Learning")
    print("=" * 55)

    print("\n[1] Generating synthetic epidemic dataset …")
    df = generate_dataset(n_samples=500)
    df.to_csv("data/sir_dataset.csv", index=False)
    print(f"    Dataset saved → data/sir_dataset.csv  ({len(df)} samples)")

    print("\n[2] Training ML models …")
    results = train_model(df)

    print("\n[3] Plotting SIR curve …")
    plot_sir_curve(beta=0.3, gamma=0.1, save_path="data/sir_curve.png")

    print("\n[4] Plotting predictions …")
    plot_predictions(results, save_path="data/predictions.png")

    print("\nDone! ✓")
