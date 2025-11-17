# Phase 0: Research Resources and Findings

## Research Context

This research builds on the concept of "Critique Markets" - using automated reviewer feedback as a resource allocation signal for AI research systems. The hypothesis proposes that treating reviewer scores as budget signals will improve novelty-adjusted accuracy and reveal domain-dependent scaling laws.

## Key Background: Kosmos AI Scientist

### Source
- **Paper**: "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv:2511.02824, Nov 2024)
- **Authors**: Describes an automated AI research system
- **Key Finding**: Linear scaling law observed - work equivalency scales linearly with computation cycles

### Kosmos Findings Relevant to This Research
1. **Linear Scaling**: Collaborators reported that scientific findings scale linearly with Kosmos cycles (up to 20 cycles tested)
2. **Accuracy**: 79.4% of statements in reports found accurate by independent scientists
3. **Work Equivalency**: Single 20-cycle run = ~6 months of human research time on average
4. **Volume**: Average run executes 42,000 lines of code, reads 1,500 papers

### Hypothesis Extension
The current research proposes that adding a "critique market" mechanism will:
- (i) Increase **novelty-adjusted** accuracy per cycle (improvement on the 79.4% baseline)
- (ii) Reveal domain-dependent saturation points earlier than naive linear extrapolation

## Related Work on Critique and Resource Allocation

### AI-Assisted Peer Review (2024-2025)
- **Finding**: LLMs can provide useful peer review feedback and predict acceptance/rejection
- **Application**: "Review Report Cards" use LLMs to analyze reviews for coverage, specificity, evidence, constructiveness
- **Relevance**: Demonstrates automated critique is feasible and correlates with quality

### Exploration-Exploitation in LLM Agents
- **WESE Framework**: Information from exploration serves as environmental prior for reasoning
- **Resource Efficiency**: Weaker models (7B params) capable of exploratory tasks
- **Challenge**: Managing tokens/communication in large-scale LLM agent scenarios
- **Attention Mechanisms**: Can allocate weights to prioritize critical data

## Proposed Experimental Approach

### Core Concept
Embed an automated "picky reviewer" that assigns scores to ideas/outputs. Treat these scores as "attention tokens" - a budget that controls how much computational resource (cycles) the AI agent allocates to:
- Reading papers
- Writing code
- Synthesizing findings

### What to Test
1. **Baseline**: Standard uniform allocation (similar to Kosmos)
   - Fixed cycles per task
   - No feedback-based prioritization

2. **Critique Market**: Reviewer-guided allocation
   - High-scoring ideas get more cycles
   - Low-scoring ideas get fewer or are abandoned
   - Dynamic resource reallocation based on feedback

3. **Metrics**:
   - **Novelty-adjusted accuracy**: Accuracy weighted by idea novelty/originality
   - **Efficiency**: Findings per unit computation
   - **Scaling behavior**: Does performance saturate? At what point?

### Datasets/Domains to Test
Since this is about domain-dependent saturation, we need multiple domains:

**Option 1: Use Existing Research Datasets**
- Kaggle datasets across domains (health, finance, social science)
- UCI ML Repository datasets
- HuggingFace datasets in different domains

**Option 2: Synthetic but Realistic**
- Generate research questions in different domains
- Use real papers from arXiv as "literature"
- Evaluate findings against ground truth from literature

**Decision**: Use Option 1 (existing datasets) with real papers from arXiv for literature search
- More realistic and credible
- Can validate against known results
- Aligns with Kosmos methodology

### Implementation Plan

**Components Needed:**
1. **Automated Reviewer/Critic**
   - LLM-based (GPT-4 or Claude) scoring system
   - Criteria: Novelty, soundness, significance, clarity
   - Returns score 0-10 and brief justification

2. **Resource Allocation Scheduler**
   - Takes reviewer scores as input
   - Allocates "cycles" (LLM calls, computation time) proportionally
   - Implements exploration-exploitation trade-off

3. **Mini Research Agent**
   - Can read papers (via web search, arXiv API)
   - Can analyze data (Python/pandas)
   - Can generate hypotheses and findings
   - Simplified version of Kosmos

4. **Evaluation Harness**
   - Tracks resource usage
   - Measures accuracy and novelty
   - Compares baseline vs critique market

### Baselines
1. **Uniform Allocation**: Equal cycles to all ideas (naive Kosmos-like)
2. **Random Allocation**: Random assignment of cycles (control for structure)
3. **Critique Market**: Reviewer-score-based allocation (proposed method)

### Evaluation Metrics
1. **Novelty-Adjusted Accuracy** = Accuracy × Novelty Score
   - Accuracy: % of correct/valid findings
   - Novelty: How unique/non-obvious (1-10 scale)

2. **Efficiency** = Valid Findings / Total Cycles Used

3. **Scaling Law Parameters**:
   - Slope: Rate of improvement with cycles
   - Saturation point: Where returns diminish
   - Domain variance: How much these differ across domains

## Resource Requirements

### Compute
- CPU only (as specified)
- Multiple LLM API calls needed

### APIs
- OpenAI GPT-4/GPT-5 OR Anthropic Claude OR OpenRouter
- arXiv API (free) for paper retrieval
- Potential cost: ~$20-50 for experiments

### Time
- 1 hour limit (3600 seconds)
- Need to keep experiments small-scale
- Focus on proof-of-concept, not full Kosmos-scale runs

### Libraries
- `openai` or `anthropic` for LLM calls
- `requests` for arXiv API
- `pandas`, `numpy` for data analysis
- `matplotlib`, `seaborn` for visualization
- `scipy` for statistical tests

## Key Decisions Made

1. **Use real LLM APIs** (not simulated agents) - per guidelines
2. **Small-scale proof of concept** - 3-5 research questions, 3 domains
3. **Existing datasets** - from Kaggle/HuggingFace
4. **Real papers** - from arXiv for literature
5. **Focus on comparative analysis** - baseline vs critique market

## Open Questions (to address in planning)

1. How to operationalize "novelty" scoring?
2. What specific allocation rule for critique market?
3. How many cycles per experiment given time constraint?
4. Which specific domains to test?
5. How to validate findings programmatically?

## Time Spent on Phase 0
- Approximately 15 minutes
- Found key resources (Kosmos paper, related work)
- Made critical design decisions
- Ready to proceed to detailed planning

---

**Proceeding to Phase 1: Detailed Planning**
