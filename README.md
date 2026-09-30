# 🧪 Bayesian A/B Test Calculator

Given two variants' conversion data, this tool computes the **Bayesian posterior** for each variant's true conversion rate, samples 100,000 times from those posteriors, and answers the question you actually want answered:

> **"What's the probability B is better than A — and by how much?"**

It also gives you a clear **ship / kill / keep testing** verdict and a self-contained HTML report with overlapping posterior distributions and the lift histogram.

## Why Bayesian instead of frequentist

Frequentist A/B testing answers *"is this difference statistically significant?"* &mdash; which is a question about the **test procedure**, not about the variants. It's also famously easy to misuse (peeking at p-values, the multiple-comparisons trap, the sample-size dance).

Bayesian A/B testing answers what you actually care about:

| Question | Bayesian answer |
|---|---|
| Is B better than A? | **P(B > A) = X%** &mdash; a direct probability |
| How much better? | Median lift, with a credible interval |
| What's the risk if I ship B and I'm wrong? | Expected loss (in absolute %) |
| Should I stop the test? | Built-in decision rule |

Easier to explain to non-statisticians, easier to act on, harder to abuse with peeking.

## What it does

1. Takes two variants' `(trials, successes)` pairs
2. Builds Beta posterior distributions: `Beta(α + successes, β + failures)` with non-informative `Beta(1, 1)` prior
3. Draws 100,000 samples from each posterior
4. Computes the difference distribution (B − A)
5. Reports `P(B > A)`, median lift, 90% credible interval, expected loss
6. Renders a verdict: **SHIP B**, **KILL B**, or **KEEP TESTING**
7. Saves a self-contained HTML report

## Quick start

```bash
git clone https://github.com/carlowenbelen/bayesian-ab-calculator.git
cd bayesian-ab-calculator
pip install -r requirements.txt

# Built-in scenarios (great for learning what the verdicts look like)
python bayesian_ab.py --list-presets
python bayesian_ab.py --preset email-headline
python bayesian_ab.py --preset landing-page    # an example that says KILL
python bayesian_ab.py --preset small-sample    # tiny sample → KEEP TESTING

# Your own data
python bayesian_ab.py \
    --a-trials 1000 --a-successes 120 \
    --b-trials 1000 --b-successes 145 \
    --a-name Original --b-name New
```

Sample terminal output:

```
══════════════════════════════════════════════════════════════════
  🧪  Bayesian A/B Test — Original vs New
══════════════════════════════════════════════════════════════════

  Original (control):    470 / 5,000  =  9.40%
  New (variant):         555 / 5,000  =  11.10%

  P(B beats A):                 99.4%
  Median absolute lift:       +1.70 pp
  Median relative lift:        +18.1%
  90% credible interval:    [+0.84, +2.55] pp
  Expected loss if ship B:    0.001%

  ► SHIP B
    P(B > A) = 99.4% ≥ 95% threshold and expected loss if wrong is
    only 0.001% (under 0.5% tolerance).
```

## CLI flags

| Flag | What it does |
|---|---|
| `--preset <name>` | Run a built-in scenario |
| `--list-presets` | Show all built-in scenarios and exit |
| `--a-trials <int>` | Control: total trials |
| `--a-successes <int>` | Control: number of conversions |
| `--b-trials <int>` | Variant: total trials |
| `--b-successes <int>` | Variant: number of conversions |
| `--a-name <str>` | Display name for control (default: "A") |
| `--b-name <str>` | Display name for variant (default: "B") |
| `--ship-threshold <0-1>` | P(B>A) needed to recommend SHIP (default: 0.95) |
| `--kill-threshold <0-1>` | P(B>A) at or below which to recommend KILL (default: 0.05) |
| `--seed <int>` | Random seed for reproducible sampling |
| `--no-report` | Terminal output only, skip the HTML report |

## How it works (the math)

A conversion rate is a coin: it produces successes with some unknown probability `p`. Given `n` trials and `k` successes, the **Beta-Binomial conjugate** gives the posterior in closed form:

```
Posterior  ~  Beta(α + k,  β + (n − k))
```

with prior `α = β = 1` (uniform on [0,1] — "I know nothing yet").

For two variants A and B:
1. `pA ~ Beta(1 + successesA, 1 + failuresA)`
2. `pB ~ Beta(1 + successesB, 1 + failuresB)`
3. Sample 100,000 pairs `(pAᵢ, pBᵢ)`
4. `P(B > A) = mean(pBᵢ > pAᵢ)`
5. `Lift distribution = (pBᵢ − pAᵢ)`

Decision rule:
- **SHIP B** if `P(B > A) ≥ 95%` AND expected loss if wrong < 0.5%
- **KILL B** if `P(B > A) ≤ 5%`
- **KEEP TESTING** otherwise

## Built-in scenarios

| Preset | Demonstrates |
|---|---|
| `email-headline` | Clear winner — produces SHIP B |
| `checkout-button` | Inconclusive — produces KEEP TESTING |
| `landing-page` | Negative result — produces KILL B |
| `small-sample` | Big apparent lift but too little data |
| `tiebreaker` | Two near-identical performers |

## Honest caveats

- **Single comparison.** This tool tests A vs B. For multivariate testing (3+ variants) you need to either run pairwise comparisons or build a multinomial-Dirichlet version (TODO).
- **Independent observations assumed.** Each trial is treated as independent. Real funnels have repeat visitors, network effects, time-of-day biases. Treat results as a starting point.
- **Stationary world assumed.** If the underlying conversion rate is changing during the test (e.g. seasonality), neither Bayesian nor frequentist methods work cleanly. Run tests in stable windows.
- **The Beta(1,1) prior is non-informative.** If you have strong prior beliefs (e.g. "we know our baseline is around 10% from history"), you can encode them by adjusting α and β in the source.
- **No multiple-testing correction.** If you're running many tests, your false-positive rate climbs. Bayesian framework helps but doesn't make this go away.

## Tech

- **Python 3.10+**
- **NumPy** for sampling and array math
- **SciPy** for Beta PDF / CDF computations
- **matplotlib** for charts
- ~430 lines, zero data-science framework dependencies

## Possible extensions

- **Multivariate testing** with Dirichlet priors
- **Sequential testing** with peek-safe stopping rules
- **Time series** view (how conversion rate evolved during the test)
- **Power analysis** — how big does the sample need to be to detect lift X?
- **Web UI** — single-page app where you paste counts and see the chart live

## License

MIT — use it, fork it, ship something.

---

Built by **Carl Owen E. Belen** &middot; companion to my [Trading Strategy Monte Carlo](https://github.com/carlowenbelen/trading-strategy-monte-carlo) and [Portfolio Optimizer](https://github.com/carlowenbelen/portfolio-optimizer) &middot; [Portfolio](https://github.com/carlowenbelen)
