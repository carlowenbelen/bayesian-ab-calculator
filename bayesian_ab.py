"""
🧪  Bayesian A/B Test Calculator
================================
Given two variants' conversion data (trials and successes), this tool computes
the **Bayesian posterior** for each variant's true conversion rate, then samples
from those posteriors to estimate:

  • P(B beats A)
  • Expected lift (and its credible interval)
  • Risk of being wrong if you ship B
  • Whether you have enough data to call it

A self-contained HTML report renders the overlapping posterior distributions
so you can *see* the uncertainty, not just read a p-value.

Why Bayesian instead of frequentist?
------------------------------------
Frequentist A/B testing answers "is this difference statistically significant?"
That's a question about the **test procedure**, not about the variants. Bayesian
A/B testing answers the question you actually care about: "what's the
**probability** B is better than A — and by how much?" Easier to explain,
easier to act on, harder to misuse.

Quick start
-----------
    pip install -r requirements.txt

    # Two variants, raw counts
    python bayesian_ab.py --a-trials 1000 --a-successes 120 \
                          --b-trials 1000 --b-successes 145

    # Run a built-in scenario
    python bayesian_ab.py --preset email-headline

Output
------
    Terminal summary + HTML report in reports/<timestamp>.html with:
      • Overlapping posterior densities for A and B
      • Histogram of the lift (B − A) from samples
      • A "ship / keep testing / kill" decision recommendation

Author
------
Carl Owen E. Belen — https://github.com/YOUR-USERNAME
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


N_SAMPLES = 100_000  # posterior samples to draw
PRIOR_ALPHA = 1.0    # uniform Beta(1,1) prior — non-informative
PRIOR_BETA = 1.0


# ============================================================
# 📐  Inputs / outputs
# ============================================================
@dataclass
class Variant:
    name: str
    trials: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def alpha(self) -> float:
        """Posterior Beta α = prior α + successes."""
        return PRIOR_ALPHA + self.successes

    @property
    def beta(self) -> float:
        """Posterior Beta β = prior β + failures."""
        return PRIOR_BETA + (self.trials - self.successes)


@dataclass
class TestResult:
    a: Variant
    b: Variant
    a_samples: np.ndarray
    b_samples: np.ndarray
    lift_samples: np.ndarray  # B − A in absolute terms

    @property
    def prob_b_beats_a(self) -> float:
        return float(np.mean(self.b_samples > self.a_samples))

    @property
    def median_lift(self) -> float:
        return float(np.median(self.lift_samples))

    @property
    def lift_p5(self) -> float:
        return float(np.percentile(self.lift_samples, 5))

    @property
    def lift_p95(self) -> float:
        return float(np.percentile(self.lift_samples, 95))

    @property
    def relative_lift(self) -> float:
        """Median (B - A) / A in percentage terms."""
        if self.a.rate == 0:
            return 0.0
        return self.median_lift / self.a.rate

    @property
    def expected_loss_if_ship_b(self) -> float:
        """Average shortfall of B vs A in worlds where A is actually better."""
        worse = (self.b_samples < self.a_samples)
        if not np.any(worse):
            return 0.0
        return float(np.mean(self.a_samples[worse] - self.b_samples[worse]))


# ============================================================
# 🎲  Sampler
# ============================================================
def run_test(a: Variant, b: Variant, seed: int | None = None) -> TestResult:
    rng = np.random.default_rng(seed)
    a_samples = rng.beta(a.alpha, a.beta, N_SAMPLES)
    b_samples = rng.beta(b.alpha, b.beta, N_SAMPLES)
    lift = b_samples - a_samples
    return TestResult(a, b, a_samples, b_samples, lift)


# ============================================================
# 🏁  Decision rule
# ============================================================
def recommend(result: TestResult,
              ship_threshold: float = 0.95,
              kill_threshold: float = 0.05,
              loss_tolerance: float = 0.005) -> tuple[str, str]:
    """Return (verdict, reason) where verdict ∈ SHIP, KILL, KEEP_TESTING."""
    p = result.prob_b_beats_a
    loss = result.expected_loss_if_ship_b

    if p >= ship_threshold and loss < loss_tolerance:
        return ("SHIP B",
                f"P(B > A) = {p:.1%} ≥ {ship_threshold:.0%} threshold and "
                f"expected loss if wrong is only {loss:.2%} (under {loss_tolerance:.1%} tolerance).")
    if p <= kill_threshold:
        return ("KILL B",
                f"P(B > A) = {p:.1%} ≤ {kill_threshold:.0%} threshold. "
                f"B is very likely worse — don't ship.")
    return ("KEEP TESTING",
            f"P(B > A) = {p:.1%} (need ≥ {ship_threshold:.0%} or ≤ {kill_threshold:.0%}). "
            f"Collect more data.")


# ============================================================
# 🎨  Charts
# ============================================================
def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def chart_posteriors(result: TestResult) -> str:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    # x-axis range that comfortably covers both posteriors
    a_lo, a_hi = stats.beta.ppf([0.001, 0.999], result.a.alpha, result.a.beta)
    b_lo, b_hi = stats.beta.ppf([0.001, 0.999], result.b.alpha, result.b.beta)
    lo = min(a_lo, b_lo)
    hi = max(a_hi, b_hi)
    pad = (hi - lo) * 0.05
    x = np.linspace(max(0, lo - pad), min(1, hi + pad), 600)

    a_pdf = stats.beta.pdf(x, result.a.alpha, result.a.beta)
    b_pdf = stats.beta.pdf(x, result.b.alpha, result.b.beta)

    ax.fill_between(x, a_pdf, alpha=0.45, color="#22d3ee", label=f"{result.a.name} posterior")
    ax.fill_between(x, b_pdf, alpha=0.45, color="#a3e635", label=f"{result.b.name} posterior")
    ax.plot(x, a_pdf, color="#22d3ee", linewidth=1.5)
    ax.plot(x, b_pdf, color="#a3e635", linewidth=1.5)

    ax.axvline(result.a.rate, color="#22d3ee", linestyle="--", alpha=0.6, linewidth=1)
    ax.axvline(result.b.rate, color="#a3e635", linestyle="--", alpha=0.6, linewidth=1)

    ax.set_title("Posterior distributions of conversion rate",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Conversion rate", color="#aaa")
    ax.set_ylabel("Density", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


def chart_lift(result: TestResult) -> str:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    lift_pct_points = result.lift_samples * 100
    ax.hist(lift_pct_points, bins=80, color="#a3e635", alpha=0.85, edgecolor="none")
    ax.axvline(0, color="white", linewidth=1.2, linestyle=":", label="Zero lift (no difference)")
    ax.axvline(result.median_lift * 100, color="#fbbf24", linewidth=2,
               label=f"Median lift: {result.median_lift * 100:+.2f} pp")
    ax.axvline(result.lift_p5 * 100, color="#fb7185", linewidth=1.5, linestyle="--",
               label=f"5th pct: {result.lift_p5 * 100:+.2f} pp")
    ax.axvline(result.lift_p95 * 100, color="#22d3ee", linewidth=1.5, linestyle="--",
               label=f"95th pct: {result.lift_p95 * 100:+.2f} pp")

    ax.set_title("Lift distribution (B − A)",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Absolute lift (percentage points)", color="#aaa")
    ax.set_ylabel("Sample frequency", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1, axis="y")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


# ============================================================
# 📄  HTML report
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bayesian A/B Test Report</title>
<style>
:root {{ --bg: #0f0f10; --card: #1a1a1a; --text: #f0f0f0; --muted: #888; --green: #a3e635; --cyan: #22d3ee; --warn: #fb7185; --gold: #fbbf24; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 3rem 2rem 5rem; }}
h1 {{ font-size: 2.5rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
.subtitle {{ color: var(--muted); margin-bottom: 2.5rem; }}
.eyebrow {{ color: var(--green); text-transform: uppercase; letter-spacing: 2px; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.4rem; }}
.verdict {{ background: var(--card); border: 2px solid #2a2a2a; border-radius: 14px; padding: 2rem 2.25rem; margin-bottom: 2.5rem; }}
.verdict.ship {{ border-color: var(--green); }}
.verdict.kill {{ border-color: var(--warn); }}
.verdict.test {{ border-color: var(--gold); }}
.verdict-badge {{ display: inline-block; padding: 0.35rem 1rem; border-radius: 50px; font-weight: 700; font-size: 0.85rem; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.8rem; }}
.verdict.ship .verdict-badge {{ background: var(--green); color: #0a2010; }}
.verdict.kill .verdict-badge {{ background: var(--warn); color: #2a0a10; }}
.verdict.test .verdict-badge {{ background: var(--gold); color: #2a1f00; }}
.verdict-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }}
.verdict-reason {{ color: var(--muted); }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
.stat {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1.25rem 1.5rem; }}
.stat-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }}
.stat-value {{ font-size: 2rem; font-weight: 700; line-height: 1; color: white; }}
.stat-value.green {{ color: var(--green); }}
.stat-value.gold {{ color: var(--gold); }}
.stat-value.warn {{ color: var(--warn); }}
.chart {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; }}
.chart img {{ width: 100%; display: block; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem; }}
th, td {{ padding: 0.85rem 1.25rem; text-align: left; border-bottom: 1px solid #222; }}
th {{ background: #161616; color: var(--green); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.4px; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
footer {{ color: var(--muted); border-top: 1px solid #222; padding-top: 1.5rem; margin-top: 3rem; font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">
  <div class="eyebrow">Bayesian A/B Test Report</div>
  <h1>{title}</h1>
  <div class="subtitle">{n_samples:,} posterior samples · Beta(1,1) prior · generated {generated_at}</div>

  <div class="verdict {verdict_class}">
    <div class="verdict-badge">{verdict}</div>
    <div class="verdict-title">{verdict_headline}</div>
    <div class="verdict-reason">{verdict_reason}</div>
  </div>

  <div class="stat-grid">
    <div class="stat"><div class="stat-label">P(B beats A)</div><div class="stat-value {pba_class}">{prob_b_beats_a:.1%}</div></div>
    <div class="stat"><div class="stat-label">Median absolute lift</div><div class="stat-value gold">{abs_lift:+.2f} pp</div></div>
    <div class="stat"><div class="stat-label">Median relative lift</div><div class="stat-value gold">{rel_lift:+.1%}</div></div>
    <div class="stat"><div class="stat-label">90% credible interval</div><div class="stat-value">{ci_low:+.2f} … {ci_high:+.2f} pp</div></div>
    <div class="stat"><div class="stat-label">Expected loss if ship B</div><div class="stat-value {loss_class}">{loss:.3%}</div></div>
  </div>

  <h2 style="margin-top: 2.5rem;">Variant data</h2>
  <table>
    <thead><tr><th>Variant</th><th class="num">Trials</th><th class="num">Successes</th><th class="num">Conversion</th><th class="num">95% credible interval</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <div class="chart"><img src="data:image/png;base64,{chart_post}"></div>
  <div class="chart"><img src="data:image/png;base64,{chart_lift}"></div>

  <footer>
    Generated by <strong>Bayesian A/B Test Calculator</strong>.<br>
    "pp" = percentage points (absolute). Posterior = Beta(α + successes, β + failures) with non-informative Beta(1,1) prior.
  </footer>
</div>
</body>
</html>
"""


def render_report(result: TestResult, verdict: str, reason: str,
                  output_path: Path | str):
    print("  → rendering charts …")
    chart_post = chart_posteriors(result)
    chart_l = chart_lift(result)

    a_lo95, a_hi95 = stats.beta.ppf([0.025, 0.975], result.a.alpha, result.a.beta)
    b_lo95, b_hi95 = stats.beta.ppf([0.025, 0.975], result.b.alpha, result.b.beta)

    rows = (
        f"<tr><td><strong>{result.a.name}</strong></td>"
        f"<td class='num'>{result.a.trials:,}</td>"
        f"<td class='num'>{result.a.successes:,}</td>"
        f"<td class='num'>{result.a.rate:.2%}</td>"
        f"<td class='num'>{a_lo95:.2%} … {a_hi95:.2%}</td></tr>"
        f"<tr><td><strong>{result.b.name}</strong></td>"
        f"<td class='num'>{result.b.trials:,}</td>"
        f"<td class='num'>{result.b.successes:,}</td>"
        f"<td class='num'>{result.b.rate:.2%}</td>"
        f"<td class='num'>{b_lo95:.2%} … {b_hi95:.2%}</td></tr>"
    )

    verdict_class = {"SHIP B": "ship", "KILL B": "kill",
                     "KEEP TESTING": "test"}.get(verdict, "test")
    pba_class = "green" if result.prob_b_beats_a >= 0.95 else (
        "warn" if result.prob_b_beats_a <= 0.05 else "")
    loss_class = "green" if result.expected_loss_if_ship_b < 0.005 else "warn"

    headlines = {
        "SHIP B": "Ship B with confidence.",
        "KILL B": "Don't ship B — it's very likely worse.",
        "KEEP TESTING": "Inconclusive — collect more data.",
    }

    html = HTML.format(
        title=f"{result.a.name} vs {result.b.name}",
        n_samples=N_SAMPLES,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        verdict_class=verdict_class,
        verdict_headline=headlines.get(verdict, verdict),
        verdict_reason=reason,
        prob_b_beats_a=result.prob_b_beats_a,
        pba_class=pba_class,
        abs_lift=result.median_lift * 100,
        rel_lift=result.relative_lift,
        ci_low=result.lift_p5 * 100,
        ci_high=result.lift_p95 * 100,
        loss=result.expected_loss_if_ship_b,
        loss_class=loss_class,
        rows=rows,
        chart_post=chart_post,
        chart_lift=chart_l,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ============================================================
# 🎯  Presets
# ============================================================
PRESETS_FILE = Path(__file__).resolve().parent / "presets.json"


def load_presets() -> dict:
    if not PRESETS_FILE.exists():
        return {}
    return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))


# ============================================================
# 🖨️  Terminal output
# ============================================================
class K:
    CY = "\033[96m"; G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    M = "\033[35m"; B = "\033[94m"; W = "\033[97m"; DIM = "\033[2m"
    BOLD = "\033[1m"; END = "\033[0m"


def c(t, color):
    return f"{color}{t}{K.END}"


def print_summary(result: TestResult, verdict: str, reason: str):
    a, b = result.a, result.b
    print()
    print(c("═" * 68, K.DIM))
    print(c(f"  🧪  Bayesian A/B Test — {a.name} vs {b.name}", K.BOLD + K.CY))
    print(c("═" * 68, K.DIM))
    print()
    print(f"  {a.name} (control):    {a.successes:>6,} / {a.trials:>6,}  =  {a.rate:.2%}")
    print(f"  {b.name} (variant):    {b.successes:>6,} / {b.trials:>6,}  =  {b.rate:.2%}")
    print()

    pba_color = K.G if result.prob_b_beats_a >= 0.95 else (
        K.R if result.prob_b_beats_a <= 0.05 else K.Y)
    print(c(f"  P(B beats A):              {result.prob_b_beats_a:>8.1%}", pba_color))
    print(f"  Median absolute lift:    {result.median_lift * 100:>+8.2f} pp")
    print(f"  Median relative lift:    {result.relative_lift:>+8.1%}")
    print(f"  90% credible interval:   "
          f"[{result.lift_p5 * 100:+.2f}, {result.lift_p95 * 100:+.2f}] pp")
    loss_color = K.G if result.expected_loss_if_ship_b < 0.005 else K.Y
    print(c(f"  Expected loss if ship B: {result.expected_loss_if_ship_b:>8.3%}",
            loss_color))
    print()
    verdict_color = {"SHIP B": K.G, "KILL B": K.R, "KEEP TESTING": K.Y}.get(verdict, K.W)
    print(c(f"  ► {verdict}", K.BOLD + verdict_color))
    print(c(f"    {reason}", K.DIM))
    print()
    print(c("═" * 68, K.DIM))


# ============================================================
# 🚀  CLI
# ============================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Bayesian A/B test calculator.")
    parser.add_argument("--preset", help="run a built-in scenario")
    parser.add_argument("--list-presets", action="store_true",
                        help="list built-in presets and exit")
    parser.add_argument("--a-trials", type=int, help="control: total trials")
    parser.add_argument("--a-successes", type=int, help="control: number of conversions")
    parser.add_argument("--b-trials", type=int, help="variant: total trials")
    parser.add_argument("--b-successes", type=int, help="variant: number of conversions")
    parser.add_argument("--a-name", default="A", help="display name for control")
    parser.add_argument("--b-name", default="B", help="display name for variant")
    parser.add_argument("--ship-threshold", type=float, default=0.95,
                        help="P(B>A) needed to recommend SHIP (default: 0.95)")
    parser.add_argument("--kill-threshold", type=float, default=0.05,
                        help="P(B>A) at or below which to recommend KILL (default: 0.05)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible sampling")
    parser.add_argument("--no-report", action="store_true",
                        help="terminal output only, skip the HTML report")
    args = parser.parse_args()

    presets = load_presets()

    if args.list_presets:
        if not presets:
            print(c("No presets found.", K.R))
            sys.exit(0)
        print(c("📋  Available presets:", K.BOLD + K.CY))
        for key, val in presets.items():
            print(f"  {c(key, K.B):<26}  {val.get('description', '')}")
            a_info = val.get("a", {})
            b_info = val.get("b", {})
            a_name = a_info.get("name", "A")
            a_str = f"{a_name}: {a_info.get('successes', 0)}/{a_info.get('trials', 0)}"
            b_name = b_info.get("name", "B")
            b_str = f"vs {b_name}: {b_info.get('successes', 0)}/{b_info.get('trials', 0)}"
            print(f"    {c(a_str, K.DIM)}  {c(b_str, K.DIM)}")
            print()
        sys.exit(0)

    # Build variants
    if args.preset:
        if args.preset not in presets:
            print(c(f"⚠️  Unknown preset '{args.preset}'. Use --list-presets.", K.R))
            sys.exit(1)
        p = presets[args.preset]
        a = Variant(p["a"]["name"], p["a"]["trials"], p["a"]["successes"])
        b = Variant(p["b"]["name"], p["b"]["trials"], p["b"]["successes"])
    else:
        if None in (args.a_trials, args.a_successes, args.b_trials, args.b_successes):
            print(c("⚠️  Provide --preset OR all of --a-trials --a-successes "
                    "--b-trials --b-successes.", K.R))
            print(c("    Run --list-presets to see options.", K.DIM))
            sys.exit(1)
        a = Variant(args.a_name, args.a_trials, args.a_successes)
        b = Variant(args.b_name, args.b_trials, args.b_successes)

    # Run
    print(c(f"  → drawing {N_SAMPLES:,} posterior samples …", K.DIM))
    result = run_test(a, b, seed=args.seed)
    verdict, reason = recommend(result, args.ship_threshold, args.kill_threshold)
    print_summary(result, verdict, reason)

    if not args.no_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = f"{a.name}-vs-{b.name}".lower().replace(" ", "-")
        out = Path("reports") / f"{timestamp}-{safe}.html"
        path = render_report(result, verdict, reason, out)
        print(c(f"  📄 Report saved: {path.absolute()}", K.G))
        print()


if __name__ == "__main__":
    main()
