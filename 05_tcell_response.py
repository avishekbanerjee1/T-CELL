import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Illustrative T-cell population ODE from the paper.")
    p.add_argument("--antigen", type=float, default=0.5, help="Normalized antigen signal A in [0,1]")
    p.add_argument("--alpha", type=float, default=0.7, help="Activation rate")
    p.add_argument("--beta", type=float, default=0.15, help="Decay rate")
    p.add_argument("--initial-t", type=float, default=1.0, help="Initial normalized T-cell population")
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--out", default="runs/tcell_response.csv")
    args = p.parse_args()
    if not 0 <= args.antigen <= 1: raise ValueError("--antigen must be between 0 and 1")
    if min(args.alpha, args.beta, args.initial_t) < 0: raise ValueError("Rates and initial T must be non-negative")
    times = np.linspace(0, args.days, int(args.days*4)+1)
    def rhs(t, y):
        T = y[0]
        return [(args.alpha * args.antigen - args.beta) * T]
    sol = solve_ivp(rhs, (0, args.days), [args.initial_t], t_eval=times)
    df = pd.DataFrame({"day": sol.t, "t_cell_population_normalized": sol.y[0],
                       "antigen_signal": args.antigen})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plt.figure(figsize=(7,4)); plt.plot(df.day, df.t_cell_population_normalized)
    plt.xlabel("Time (days)"); plt.ylabel("Normalized T-cell population")
    plt.title("Illustrative T-cell response ODE"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out.with_suffix(".png"), dpi=160); plt.close()
    print(f"Growth coefficient alpha*A - beta = {args.alpha*args.antigen-args.beta:.4f}")
    print(f"Saved CSV to {out.resolve()}")
    print("This simplified ODE is not clinically calibrated and does not predict real treatment success.")

if __name__ == "__main__":
    main()
