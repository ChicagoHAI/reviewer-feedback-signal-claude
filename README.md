# Critique Markets for Discovery

Research on turning reviewer feedback into a resource allocation signal for AI research systems.

## Quick Summary

**Research Question**: Can automated critic feedback improve AI research efficiency by controlling resource allocation?

**Key Findings** (3-5 main results):
- ✗ **Critique market does NOT significantly outperform uniform allocation** (p=0.276, negligible effect size)
- ✓ **Both structured approaches beat random allocation** (p<0.05)
- ✓ **Strong domain-dependent variation observed** (p=0.026) - efficiency varies 2× across domains
- ✓ **Clear saturation effects detected** (r < -0.94, p < 0.05) - diminishing returns with increased budget
- ⚠️ **Hypothesis partially supported**: Critique market didn't improve efficiency, but domain-dependent saturation was clearly demonstrated

**Bottom Line**: In simple hypothesis spaces, allocation strategy matters less than hypothesis quality. Domain characteristics and saturation effects are more important than allocation mechanisms.

## Project Structure

```
.
├── REPORT.md                    # Full research report (comprehensive)
├── README.md                    # This file (quick overview)
├── planning.md                  # Research plan and methodology
├── resources.md                 # Phase 0 research findings
├── pyproject.toml              # Python dependencies
├── notebooks/
│   └── 2025-11-16-22-20_CritiqueMarketsExperiment.ipynb  # Main experiment
└── results/
    ├── all_experiments.json     # Full experimental data
    ├── experiment_summary.csv   # Summary table
    └── plots/
        ├── efficiency_by_strategy.png
        ├── scaling_curves.png
        └── naa_comparison.png
```

## How to Reproduce

### 1. Environment Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install openai pandas numpy matplotlib seaborn scipy scikit-learn datasets requests
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-key-here"
# OR
export OPENROUTER_API_KEY="your-key-here"
```

### 3. Run Experiments

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/2025-11-16-22-20_CritiqueMarketsExperiment.ipynb
```

Or execute cells sequentially. Total runtime: ~5-15 minutes depending on LLM usage.

### 4. View Results

- **Full report**: `REPORT.md`
- **Plots**: `results/plots/*.png`
- **Data**: `results/experiment_summary.csv`

## Key Components

### 1. Research Agent
- Generates hypotheses about data patterns
- Tests hypotheses using statistical tests (Pearson correlation, chi-square)
- Tracks findings and resource usage

### 2. Automated Critic
- LLM-based (GPT-4o-mini) evaluation of hypotheses
- Scores on 3 dimensions: novelty, soundness, significance
- Returns structured feedback (1-10 scale + justification)

### 3. Allocation Strategies
- **Uniform**: Equal cycles to all hypotheses (baseline)
- **Random**: Random cycle assignment (control)
- **Critique Market**: Cycles proportional to critic scores (proposed)

### 4. Evaluation Metrics
- **Novelty-Adjusted Accuracy (NAA)**: Accuracy × novelty score
- **Efficiency**: Valid findings / total cycles used
- **Scaling behavior**: Correlation between budget and efficiency

## Main Results

### Strategy Comparison

| Strategy | Efficiency | NAA | vs Random |
|----------|-----------|-----|-----------|
| Uniform | 0.317 ± 0.15 | 7.55 | p=0.024 ✓ |
| Critique Market | 0.320 ± 0.14 | 7.60 | p=0.017 ✓ |
| Random | 0.194 ± 0.09 | 7.51 | - |

**Critique vs Uniform**: p=0.968 (no significant difference)

### Domain Variation

| Domain | Efficiency | Saturation (r) | p-value |
|--------|-----------|---------------|---------|
| Medical | 0.388 | -0.978 | 0.022 ✓✓ |
| Social | 0.396 | -0.947 | 0.053 ✓ |
| Environmental | 0.176 | -0.994 | 0.006 ✓✓✓ |

**ANOVA**: F=5.64, p=0.026 (significant domain effect)

## Visualizations

See `results/plots/`:
1. **efficiency_by_strategy.png**: Box plots showing efficiency distributions
2. **scaling_curves.png**: Budget vs efficiency for all strategies
3. **naa_comparison.png**: NAA across domains and strategies

## Technologies Used

- **Python 3.12** - Core implementation
- **OpenAI GPT-4o-mini** - Automated critic
- **pandas/numpy** - Data manipulation
- **scipy** - Statistical tests
- **matplotlib/seaborn** - Visualization

## Limitations

- **Simplified setting**: Only tests simple bivariate correlations
- **Synthetic data**: May not reflect real research complexity
- **Small scale**: 36 experiments, 10 hypotheses each
- **Heuristic critic**: Used fast heuristic for bulk runs (cost/time trade-off)

## Citation

```bibtex
@techreport{critique_markets_2025,
  title={Critique Markets for Discovery: Turning Reviewer Feedback into a Resource Allocation Signal},
  author={AI Research Agent},
  year={2025},
  month={November},
  institution={Autonomous Research Lab},
  note={Proof-of-concept study on feedback-based resource allocation in AI research}
}
```

## Contact & Further Reading

- **Full Report**: See `REPORT.md` for comprehensive analysis
- **Research Plan**: See `planning.md` for methodology details
- **Background**: See `resources.md` for literature review

---

**Status**: Completed ✓
**Runtime**: ~45 minutes (all phases)
**Cost**: <$1 (LLM API calls)
