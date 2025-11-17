# Research Plan: Critique Markets for Discovery

## Research Question

**Primary Question**: Does turning reviewer feedback into a market signal that controls resource allocation (exploration-exploitation and cycle allocation) increase novelty-adjusted accuracy per cycle and reveal domain-dependent scaling laws that differ from naive linear extrapolation?

**Specific Sub-Questions**:
1. Does reviewer-guided resource allocation improve efficiency (findings per cycle) compared to uniform allocation?
2. Does it improve novelty-adjusted accuracy compared to uniform allocation?
3. Do different domains show different scaling behaviors and saturation points?
4. Can we detect saturation earlier than linear extrapolation would predict?

## Background and Motivation

### Why This Matters

The Kosmos AI Scientist (Nov 2024) demonstrated **linear scaling** of scientific findings with computational cycles. However, this is a naive uniform allocation strategy - like giving every research direction equal resources. In real science, researchers prioritize promising leads and abandon dead ends based on intermediate feedback.

**This research asks**: Can we improve on Kosmos's linear scaling by introducing a "market mechanism" where an automated critic assigns value ("attention tokens") to ideas, and computational resources flow toward high-value directions?

### Gap This Fills

1. **Efficiency**: Current AI scientists treat all research directions equally (wasteful)
2. **Exploration-Exploitation**: No principled way to balance novel vs. safe ideas
3. **Domain Generalization**: Unknown whether one strategy works across domains
4. **Saturation Detection**: Linear scaling may overestimate long-term returns

### Expected Impact

- **Scientific**: New understanding of AI research scaling laws with feedback
- **Practical**: More efficient AI research agents (less compute waste)
- **Theoretical**: Insights into domain-dependent learning dynamics

## Hypothesis Decomposition

### Primary Hypothesis (H1)
**Critique Market allocation increases novelty-adjusted accuracy per cycle compared to uniform allocation**

- **Operationalization**:
  - Independent Variable: Allocation strategy (uniform vs. critique-market)
  - Dependent Variable: (Accuracy × Novelty) / Cycles_used
  - Success: Critique-market shows statistically significant improvement (p < 0.05)

### Secondary Hypothesis (H2)
**Critique Market reveals saturation points earlier and more clearly than linear extrapolation**

- **Operationalization**:
  - Measure performance vs. cycles for both strategies
  - Fit power-law model: Performance = a × Cycles^b
  - If b < 1 (sublinear), saturation is present
  - Success: Critique-market shows earlier/clearer saturation in at least 2/3 domains

### Tertiary Hypothesis (H3)
**Scaling behavior is domain-dependent (varies across research domains)**

- **Operationalization**:
  - Run experiments in 3 domains with different characteristics
  - Measure scaling exponent (b) and saturation point for each
  - Success: Significant variance in scaling parameters across domains (ANOVA p < 0.05)

## Proposed Methodology

### High-Level Approach

We'll build a **minimal viable AI research agent** that:
1. Starts with a research question and small dataset
2. Generates hypotheses about patterns in the data
3. Has an **automated critic** score each hypothesis
4. Allocates computational "cycles" based on critic scores
5. Pursues high-scoring hypotheses more deeply
6. Generates findings and evaluates accuracy/novelty

This simulates a simplified Kosmos-like system with added feedback mechanism.

**Justification**:
- Simplified enough to run in 1 hour
- Complex enough to test the core hypothesis
- Uses real LLMs (not simulations) per guidelines
- Tests across multiple domains (generalization)

### Experimental Steps

#### Step 1: Domain and Dataset Selection (10 min)
**What**: Select 3 diverse research domains with different characteristics
**Why**: Test domain-dependency hypothesis (H3)
**How**:
- Choose from Kaggle/HuggingFace datasets
- Criteria: Different domain, ~1000-5000 rows, interpretable features
- Domains to target:
  - **Structured/Tabular** (e.g., medical diagnosis, credit scoring)
  - **Behavioral/Social** (e.g., user preferences, social networks)
  - **Physical/Natural** (e.g., environmental data, biology)

#### Step 2: Implement Research Agent Core (15 min)
**What**: Build agent that can generate and test hypotheses
**Why**: Need working system to compare allocation strategies
**How**:
- Component A: Hypothesis Generator (LLM generates hypotheses from data)
- Component B: Hypothesis Tester (Python code tests hypothesis on data)
- Component C: Accuracy Evaluator (Check if finding is statistically valid)

**Rationale**: Keep it simple - focus on correlation/pattern discovery, not complex ML

#### Step 3: Implement Automated Critic (10 min)
**What**: LLM-based scoring system for hypotheses
**Why**: Core innovation - this is the "market signal"
**How**:
- Input: Hypothesis + supporting evidence
- Output: Score (0-10) on multiple dimensions:
  - **Novelty** (1-10): How surprising/non-obvious?
  - **Soundness** (1-10): How statistically valid?
  - **Significance** (1-10): How practically important?
- Use GPT-4 or Claude with structured prompt
- Return average score + brief justification

**Rationale**: Multi-dimensional scoring captures research quality better than single metric

#### Step 4: Implement Allocation Strategies (10 min)
**What**: Different rules for assigning cycles to hypotheses
**Why**: Test H1 - does smart allocation beat uniform?
**How**:

**Baseline 1 - Uniform Allocation**:
- All hypotheses get equal cycles (e.g., 3 cycles each)
- Simple, fair, but ignores quality

**Baseline 2 - Random Allocation**:
- Random assignment of cycles
- Controls for structure vs. randomness

**Proposed - Critique Market Allocation**:
- Cycles ∝ Critic Score
- High-scoring hypotheses get more cycles
- Low-scoring (<threshold) get abandoned
- Formula: cycles_i = floor((score_i / sum(scores)) × total_budget)

**Rationale**: Critique market should focus resources on promising directions

#### Step 5: Run Comparative Experiments (20 min)
**What**: Execute all 3 strategies across all 3 domains
**Why**: Generate data to test all hypotheses
**How**:
- For each domain:
  - For each strategy:
    - Run agent for varying cycle budgets (5, 10, 15, 20 cycles)
    - Track: hypotheses generated, scores, findings, accuracy, cycles used
    - Measure novelty-adjusted accuracy
- Total: 3 domains × 3 strategies × 4 budget levels = 36 runs

**Rationale**: Multiple budget levels needed to observe scaling behavior

#### Step 6: Statistical Analysis (10 min)
**What**: Test hypotheses with appropriate statistical tests
**Why**: Rigorous validation of claims
**How**:
- **H1 Test**: Paired t-test comparing critique-market vs. uniform on novelty-adjusted accuracy per cycle
- **H2 Test**: Fit power-law models, compare saturation points
- **H3 Test**: ANOVA on scaling parameters across domains
- Calculate effect sizes (Cohen's d)
- Generate confidence intervals

**Rationale**: Standard statistical practice for comparative studies

#### Step 7: Visualization and Reporting (10 min)
**What**: Create plots and document findings
**Why**: Communicate results clearly
**How**:
- Scaling curves (performance vs. cycles) for each strategy
- Bar charts comparing efficiency across strategies
- Domain-specific saturation point visualization
- Save all results to structured JSON

**Rationale**: Visual evidence complements statistical tests

### Baselines

1. **Uniform Allocation** (Primary baseline)
   - Industry standard (similar to Kosmos)
   - Easy to implement
   - Strong baseline - hard to beat

2. **Random Allocation** (Control)
   - Tests whether *any* structure helps
   - If critique-market ≈ random, our mechanism doesn't work
   - If critique-market > random, some signal present

3. **Oracle Allocation** (Optional, if time permits)
   - Use ground-truth accuracy to allocate (cheating)
   - Upper bound on performance
   - Shows headroom for improvement

### Evaluation Metrics

#### Primary Metrics

**1. Novelty-Adjusted Accuracy (NAA)**
- **What**: Accuracy weighted by novelty
- **Why**: Captures both correctness and value
- **Formula**: NAA = (Σ accuracy_i × novelty_i) / N
- **Interpretation**: Higher is better; balances safe vs. novel findings

**2. Efficiency (Findings per Cycle)**
- **What**: Valid findings divided by cycles consumed
- **Why**: Measures resource efficiency
- **Formula**: Efficiency = Valid_Findings / Total_Cycles
- **Interpretation**: Higher is better; like ROI for computation

#### Secondary Metrics

**3. Scaling Exponent (b)**
- **What**: Power-law fit parameter
- **Why**: Characterizes diminishing returns
- **Formula**: Performance = a × Cycles^b
- **Interpretation**: b=1 is linear; b<1 is sublinear (saturation); b>1 is superlinear (unlikely)

**4. Saturation Point**
- **What**: Cycle count where improvement drops below threshold
- **Why**: Practical stopping criterion
- **Formula**: First point where marginal_gain < 0.05 × initial_gain
- **Interpretation**: Earlier saturation = faster resource reallocation

#### Auxiliary Metrics

- **Accuracy**: % of statistically valid findings (p < 0.05)
- **Novelty**: Average novelty score (1-10)
- **Diversity**: Number of unique hypothesis types explored

### Statistical Analysis Plan

#### Tests to Perform

**Primary Analysis**:
1. **Paired t-test**: Critique-market vs. Uniform on NAA/cycle
   - H0: No difference in NAA/cycle
   - H1: Critique-market > Uniform
   - α = 0.05 (one-tailed)
   - Effect size: Cohen's d

2. **ANOVA**: Scaling exponents across domains
   - H0: No variance in exponents across domains
   - H1: Significant variance
   - α = 0.05
   - Post-hoc: Tukey HSD if significant

**Secondary Analysis**:
3. **Bootstrap confidence intervals**: For saturation points
4. **Permutation test**: Validate t-test if normality violated

#### Corrections for Multiple Comparisons
- Bonferroni correction if testing >3 comparisons
- Report both raw and corrected p-values

#### Power Analysis
- Minimum detectable effect size: d = 0.8 (large effect)
- Required sample size: ~12 observations per group
- Achieved with: 3 domains × 4 budget levels = 12 data points

## Expected Outcomes

### If H1 is Supported (Critique-Market Wins)
- Novelty-adjusted accuracy per cycle significantly higher (p < 0.05)
- Effect size moderate to large (d > 0.5)
- **Interpretation**: Feedback mechanisms improve AI research efficiency
- **Implication**: Worth integrating into real AI research systems

### If H1 is Refuted (No Difference or Worse)
- No significant difference, or critique-market worse
- **Interpretation**: Either (a) critic is poor, or (b) uniform is surprisingly good
- **Implication**: Need better critic or different allocation rule

### If H2 is Supported (Saturation Detected)
- Power-law exponent b < 1 in critique-market
- Saturation point visible in plots
- **Interpretation**: Returns diminish predictably
- **Implication**: Can optimize stopping criteria

### If H3 is Supported (Domain Variance)
- ANOVA shows significant variance (p < 0.05)
- Different domains show different scaling exponents
- **Interpretation**: One-size-fits-all doesn't work
- **Implication**: Need domain-adaptive strategies

## Timeline and Milestones

**Total Time Budget**: 60 minutes (3600 seconds)

### Breakdown
- ✓ Phase 0 - Research & Planning: 25 min (DONE)
- Phase 2 - Setup & Implementation: 25 min
  - Environment setup: 5 min
  - Dataset loading: 5 min
  - Agent implementation: 10 min
  - Critic implementation: 5 min
- Phase 3-4 - Experiments: 20 min
  - Run 36 experiments: 15 min (~25 sec each)
  - Buffer for errors: 5 min
- Phase 5 - Analysis: 10 min
  - Statistical tests: 5 min
  - Visualization: 5 min
- Phase 6 - Documentation: 10 min
  - Write REPORT.md: 7 min
  - Write README.md: 3 min

### Critical Path
1. Get basic agent working → Can't test hypotheses without this
2. Get critic working → Core innovation depends on this
3. Run at least baseline vs. critique-market → Minimum viable test
4. Get one domain fully analyzed → Proof of concept

### Contingency Plans
- **If time runs short**: Drop random baseline and oracle, focus on uniform vs. critique
- **If experiments fail**: Simplify domains to toy problems
- **If critic is poor**: Use simpler heuristic (e.g., statistical p-value as score)
- **If no scaling observed**: Report null result honestly

## Potential Challenges

### Challenge 1: Critic Quality
**Problem**: LLM critic may be inconsistent or poorly calibrated
**Mitigation**:
- Use structured prompting with rubric
- Run critic multiple times, average scores
- Validate critic scores against ground truth on small sample

### Challenge 2: Time Constraints
**Problem**: 60 minutes is tight for 36 experiments + LLM calls
**Mitigation**:
- Use fast model (GPT-4o-mini or Claude Haiku) for hypothesis generation
- Use smarter model (GPT-4 or Claude Sonnet) only for critic
- Cache LLM responses to avoid re-computation
- Parallelize where possible

### Challenge 3: Defining "Novelty" Objectively
**Problem**: Novelty is subjective and hard to validate
**Mitigation**:
- Use proxy: inverse correlation with common patterns
- LLM scoring with clear rubric
- Focus on relative novelty (ranking) not absolute

### Challenge 4: Validating Findings
**Problem**: How to know if a finding is "correct" without human expert?
**Mitigation**:
- Use statistical validity (p-value < 0.05) as proxy for accuracy
- For known datasets, check against published results
- Report confidence intervals and limitations

### Challenge 5: Small Sample Size
**Problem**: 36 runs may not have enough power
**Mitigation**:
- Use within-subjects design (same dataset across strategies)
- Report effect sizes and confidence intervals, not just p-values
- Acknowledge as limitation
- Position as proof-of-concept, not definitive study

## Success Criteria

### Minimum Viable Success
✓ Implementation works (agent generates hypotheses, critic scores them)
✓ At least one domain tested with baseline and critique-market
✓ Some measurable difference in performance (even if not significant)
✓ Complete documentation with honest reporting

### Target Success
✓ All 3 domains tested
✓ Statistically significant improvement in at least one metric (p < 0.05)
✓ Evidence of scaling behavior (can fit power-law curve)
✓ Clear visualization of results

### Stretch Success
✓ All hypotheses tested rigorously
✓ Strong effect sizes (d > 0.8)
✓ Clear domain-dependent patterns
✓ Identified saturation points
✓ Reproducible code with documentation

## Measurement Plan

### What to Log for Each Experiment Run

```python
{
  "experiment_id": "domain1_uniform_budget10",
  "domain": "medical",
  "strategy": "uniform",
  "budget": 10,
  "hypotheses_generated": [...],
  "critic_scores": [...],
  "cycles_allocated": [...],
  "findings": [...],
  "accuracy": 0.75,
  "novelty_avg": 6.2,
  "novelty_adjusted_accuracy": 4.65,
  "valid_findings": 6,
  "total_cycles": 10,
  "efficiency": 0.6,
  "timestamp": "2025-11-16T...",
  "llm_calls": 15,
  "cost_usd": 0.23
}
```

### Aggregation
- Aggregate across budget levels for scaling curves
- Aggregate across domains for variance analysis
- Aggregate across strategies for comparisons

## Reproducibility Plan

### Random Seeds
- Set seed=42 for all random operations
- Set temperature=0 for deterministic LLM responses (where possible)
- Document when temperature>0 used

### Versioning
- Log exact model versions (gpt-4-turbo-2024-04-09, etc.)
- Log library versions (openai==1.x.x)
- Save full prompts used for critic

### Data
- Save raw datasets used
- Save all LLM responses
- Save all experiment logs

### Code
- Well-commented notebook
- Clear section headers
- Reproducible from top to bottom

## Alternatives Considered and Rejected

### Alternative 1: Simulate Critic Instead of Using LLM
**Why rejected**: Guidelines explicitly say use real LLMs, not simulations
**Trade-off**: Real LLMs are slower and cost money, but scientifically valid

### Alternative 2: Use One Domain Only
**Why rejected**: Can't test domain-dependency (H3) with one domain
**Trade-off**: Three domains takes 3× longer, but necessary for completeness

### Alternative 3: Use Complex ML Tasks
**Why rejected**: Too time-consuming to implement in 60 min
**Trade-off**: Simpler correlation/pattern discovery less impressive but feasible

### Alternative 4: Manual Novelty Scoring
**Why rejected**: Defeats purpose of automation, introduces bias
**Trade-off**: LLM scoring may be noisy, but it's systematic

## Ethical Considerations

### Research Integrity
- Report negative results honestly
- No p-hacking or selective reporting
- Acknowledge limitations clearly

### Resource Usage
- Estimated cost: $20-50 for LLM calls
- Within budget constraints
- Document actual cost in final report

### Generalizability Claims
- This is a proof-of-concept, not definitive
- Small scale limits generalization
- Position findings as preliminary

---

## Next Steps: Implementation

With this plan in hand, I will now:
1. Set up the Python environment
2. Load datasets
3. Implement the research agent
4. Implement the critic
5. Run experiments
6. Analyze results
7. Document findings

**Proceeding immediately to Phase 2: Implementation**
