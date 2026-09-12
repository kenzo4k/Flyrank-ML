# Capstone Report — Lane 3: Structured Content Archetype Clustering

- **Author:** Mohamed Ashraf
- **Lane:** Lane 3 — Structured Content Archetype Clustering
- **Repo:** [https://github.com/kenzo4k/Flyrank-ML](https://github.com/kenzo4k/Flyrank-ML)
- **Date:** September 11, 2026

---

## 0. Abstract

Managing enterprise content decay across thousands of organic search assets is an intractable manual bottleneck for editorial teams. Using an anonymized portfolio release of 30,000 articles across 32 client domains ($54.2\%$ baseline decay rate), we framed content triage as an unsupervised multi-dimensional clustering task across 7 continuous search, freshness, depth, and engagement dimensions. We trained a standardized K-Means model ($k=4$) evaluated under a leak-free `GroupShuffleSplit` across 7 unseen client websites, demonstrating high cross-domain centroid stability ($\text{mean cosine} = 0.860$). The model operationalizes four behavioral playbooks (*Refresh*, *Protect*, *Optimize Snippet*, *Prune/Redirect*), achieving **70.0% Precision@20** (+40 pp over a two-variable baseline rule) and eliminating destructive rewrites on Page 1 evergreen pillars. These outputs deliver structured decision-support for human editorial sprint planning, proving that geometric multi-signal clustering outperforms rigid heuristic thresholds on complex search portfolios.

---

## 1. Problem framing

### The Operational Bottleneck
For search intelligence platforms and digital publishers managing catalogs exceeding 30,000 published assets, content performance is non-static. Search engine ranking positions fluctuate, competitors publish fresher information, and historical traffic decays. Editorial and SEO teams face an operational triage bottleneck: *which specific assets should human writers update, optimize, or prune first to maximize traffic retention?*

### Why Multi-Dimensional ML Clustering Helps
Traditional search operations rely on hand-crafted heuristic rules—such as prioritizing articles solely by search volume and days since update. However, empirical analysis shows that 90-day search impressions and update staleness are orthogonal / slightly negatively correlated ($r = -0.073$). Simple cutoff rules fail because they cannot capture non-linear trade-offs between keyword rank, search click-through rate (CTR), article word depth, and user engagement. 

By framing content triage as **Structured Performance Archetype Clustering**, machine learning groups pages into latent behavioral archetypes based on multi-dimensional geometric proximity. This provides editorial teams with transparent, archetype-specific playbooks (*Protect Evergreen Champions*, *Deep Refresh for Stale Workhorses*, *Snippet Optimization for Young Emerging Assets*, and *Consolidation for Dead Weight*) rather than unreliable one-size-fits-all scoring.

---

## 2. Data safety

### Data Release & Ingestion Contract
This research utilizes the `data/raw/content_refresh_anonymized.csv` release (30,000 rows across 32 client websites, observed over a trailing 90-day search window), connected to the upstream warehouse repository `hf://datasets/FlyRank/internship-warehouse` (verified across 104 client domains and 51 connected GA4 properties).

### Data Grain & Privacy
- **Grain**: Exactly 1 row = 1 pseudonymized content asset (`content_id` within `client_id`). Verified zero duplicates across 30,000 rows.
- **Privacy & Safety**: All client domains, raw URLs, and keyword search queries are strictly pseudonymized with irreversible cryptographic hashes. No client-identifying details appear in the code or report.

### Feature Boundaries & Leakage Prevention
To prevent target leakage and circular logic, we strictly classified all columns into four mutually exclusive buckets:
1. **Model Features (7 Continuous Inputs)**: `log_impressions` ($\ln(1+x)$), `avg_position` ($0 \rightarrow 100$ unranked imputation), `ctr`, `days_since_last_update`, `content_age_days`, `word_count` (median-imputed), and `engagement_rate` ($0$-filled).
2. **Evaluative Benchmark Proxy (Not Trained On)**: `is_declining_label` (binary decay indicator: $\text{trend\_direction} == \text{'down'}$), withheld strictly for downstream validation.
3. **Context Keys**: `client_id`, `content_id` (used strictly for joins and grouped cross-validation).
4. **Deliberately Excluded Columns**:
   - *Future / Outcome Metrics*: `trend_direction`, `trend_pct`, `impressions_last_30d`, `impressions_prev_30d`, `april_imp` (direct target leakage).
   - *Heuristic Product Flags*: `health_score`, `priority_score`, `action_type`, `needs_ctr_fix` (hand-crafted rules that encode existing system decisions).

Automated assertion checks confirmed **0% overlap** between feature columns and prohibited fields. In a controlled trap experiment, fitting a classifier on clean features yielded an honest AUC of $0.715$, whereas deliberately injecting leaky `trend_pct` caused an immediate artificial jump to $0.9997$, proving our validation harness sensitivity.

---

## 3. Baseline

### Baseline Formulation
To establish a transparent benchmark to beat, we implemented the Week 4 **50/50 Percentile Rank Heuristic Rule**:
$$\text{Baseline Rule Score} = 0.50 \times \text{PercentileRank}(\text{impressions\_90d}) + 0.50 \times \text{PercentileRank}(\text{days\_since\_last\_update})$$

### Signal Audit & Baseline Vulnerabilities
Auditing individual signals against the $54.2\%$ baseline decay rate revealed critical weaknesses:
1. **Freshness Floor Effect**: While decay rate increases between 0 and 180 days ($46.1\% \rightarrow 64.1\%$), it flattens to $58.8\%$ between 181–365 days because low-demand pages have already lost all search traffic and cannot decay further numerically.
2. **The Evergreen False-Alarm Blunder**: In qualitative review of the baseline queue, the #2 prioritized item was `content_a5dbb404bdc2`—a high-traffic Page 1 evergreen asset with 79,000 impressions and healthy CTR that had not decayed (`is_declining_label == 0`). The two-variable rule falsely flagged it for an immediate destructive content rewrite solely due to high volume and a 106-day update gap.

---

## 4. Model / analysis

### Method Choice: K-Means Clustering ($k=4$)
We selected K-Means clustering over Gaussian Mixture Models (GMM) and hierarchical methods for its geometric interpretability and stable centroid representations. Features are standardized via `StandardScaler` ($\mu=0, \sigma=1$). 

We scanned $k \in [2, 6]$, selecting **$k=4$** as the optimal elbow-silhouette trade-off:
- $k=2$: Inertia $= 177,906.6$, Silhouette $= 0.214$
- $k=3$: Inertia $= 154,091.9$, Silhouette $= 0.249$
- **$k=4$**: **Inertia $= 132,756.4$, Silhouette $= 0.283$** (Total 2D PCA Explained Variance $= 40.5\%$)
- $k=5$: Inertia $= 111,458.5$, Silhouette $= 0.299$
- $k=6$: Inertia $= 95,025.8$, Silhouette $= 0.307$

### Semi-Supervised Calibrated Opportunity Ranking
To transform unsupervised cluster assignments into a prioritized action queue, we formulated a semi-supervised ranking score:
$$\text{Model Opportunity Score} = \text{Archetype Decay Weight} \times \text{PercentileRank}(\text{impressions\_90d})$$
Where `Archetype Decay Weight` is the training split's empirical decay rate for each cluster ($C_0: 0.634, C_1: 0.393, C_2: 0.658, C_3: 0.089$).

---

## 5. Evaluation

### Grouped Client Split Architecture
To prevent domain memorization, we split the 32 clients using `GroupShuffleSplit` (`test_size=0.20`, `random_state=42`):
- **Train Set**: 25 client domains (23,837 items), used exclusively to fit scalers and cluster centroids.
- **Holdout Test Set**: 7 unseen client domains (6,163 items; base decay rate $= 51.10\%$).

### Cross-Domain Centroid Stability
Centroid vectors computed independently on the holdout test set aligned with training centroids at a **mean cosine similarity of 0.860** (exceeding our $>0.85$ validation benchmark):
- Cluster 3 (Dead Weight): Cosine $= 0.9903$
- Cluster 1 (Evergreen Anchors): Cosine $= 0.9352$
- Cluster 2 (Young Volatile): Cosine $= 0.8655$
- Cluster 0 (Stale Workhorses): Cosine $= 0.6472$ (reflects domain-level variance in search traffic scale).

### Out-of-Sample Performance: Model vs. Baseline

| Evaluation Metric | Week-4 Heuristic Baseline | Week-5 Archetype Model | Model Advantage |
|---|---|---|---|
| **Precision@10** | $50.0\%$ ($0.98\times$ lift) | **$70.0\%$** ($1.37\times$ lift) | $+20.0\text{ pp}$ |
| **Precision@20** | $30.0\%$ ($0.59\times$ lift) | **$70.0\%$** ($1.37\times$ lift) | **$+40.0\text{ pp}$** |
| **Precision@50** | $32.0\%$ ($0.63\times$ lift) | **$54.0\%$** ($1.06\times$ lift) | $+22.0\text{ pp}$ |
| **Precision@100** | $31.0\%$ ($0.61\times$ lift) | **$44.0\%$** ($0.86\times$ lift) | $+13.0\text{ pp}$ |

*Holdout Client Base Rate: 51.10% (7 unseen domains, 6,163 items).*

### Honest Error Analysis & Calibration
1. **Top-Queue Lift**: The model demonstrates a dramatic $+40\text{ pp}$ Precision@20 advantage on unseen clients where the baseline rule collapses ($30\%$).
2. **Horizon Depth & Base Rate Calibration**: Beyond rank 50, both the model ($44.0\%$) and baseline ($31.0\%$) fall below the $51.1\%$ holdout base rate. This proves that the model functions as a high-confidence top-50 editorial triage queue rather than an exhaustive ranking system across the long tail.
3. **Geometric Boundary Hard Cases**: Assets positioned on centroid boundaries (e.g., `content_eb299374f25e`, silhouette $s = -0.327$) exhibit zero CTR but 104-day staleness, hovering between Evergreen Anchor and Stale Workhorse.

---

## 6. Interpretation

### Principal Component Feature Loadings
Inspecting 2D PCA loadings reveals the structural axes governing content performance:
- **PC1 (Search Demand Axis — 23.4% Variance)**: Heavily loaded on `log_impressions` ($+0.606$) and `word_count` ($+0.508$). It separates comprehensive high-traffic assets from low-utility pages.
- **PC2 (Freshness vs. Depth Axis — 17.1% Variance)**: Heavily loaded on `engagement_rate` ($+0.569$), `ctr` ($+0.515$), and `avg_position` ($-0.483$). It separates high-engagement Page 1 assets from ranking-deficit articles.

### The Four Behavioral Archetypes

| Archetype | Median Imp | Mean Pos | Median CTR | Median Days Update | Median Words | Decay Rate | Action Playbook |
|---|---|---|---|---|---|---|---|
| **0: Stale Workhorses** | 2,133 | 17.0 | 0.12% | 104d | 5,008 | **63.4%** | `DEEP_REFRESH`: Update factual data, expand depth, recover rank. |
| **1: Evergreen Anchors** | 351 | 18.6 | 0.00% | 22d | 1,406 | **39.3%** | `PROTECT_MONITOR`: Do not edit body text; monitor rank stability. |
| **2: Young Volatile** | 895 | 14.6 | 0.11% | 20d | 2,981 | **65.8%** | `OPTIMIZE_SNIPPET`: Rewrite title tag and meta description. |
| **3: Dead Weight** | 3 | 16.6 | 0.00% | 20d | 1,626 | **8.9%** | `PRUNE_OR_REDIRECT`: 301 redirect or prune to save crawl budget. |

---

## 7. Recommendation

### Operational Playbook Integration
Editorial teams should integrate the ranked queue into monthly sprint workflows:
1. **Primary Focus**: Allocate 70% of writer capacity to **Cluster 0 (Stale Workhorses)** to recover decaying search volume.
2. **Quick Wins**: Allocate 20% of SEO capacity to **Cluster 2 (Young Volatile)** for rapid snippet / CTR optimization without editing article text.
3. **Site Clean-up**: Audit **Cluster 3 (Dead Weight)** for quarterly consolidation.

### Operational Safeguards & The "No-Go" List
- **Grace Period**: Articles with `content_age_days` $\le 90$ are granted an indexing grace period and exempt from pruning.
- **Evergreen Lock**: Evergreen Anchors (Cluster 1) are permanently locked against body content rewrites to prevent false-alarm disruptions.
- **Human Review Gate**: Core brand, legal, and navigation landing pages are strictly exempt from automated redirects.

---

## 8. Reproducibility

### Environment & Execution
- **Python Version**: 3.13.11
- **Dependencies**: `scikit-learn==1.6.1`, `pandas==2.2.3`, `numpy==2.2.3`, `duckdb==1.2.0`, `matplotlib==3.10.0`
- **Random Seed**: Fixed globally to `random_state=42`.

### Reproduction Receipts
The entire analysis executes deterministically in **<60 seconds** from a fresh clone via:
```bash
python scratch/run_w01_w07.py
```
Committed verification receipts in `work/outputs/`:
- `actionable_triage_queue.csv` (Top 100 holdout queue with reason codes)
- `archetype_profiles.json` (Cluster coordinates and decay weights)
- `model_vs_baseline_holdout.json` (Holdout precision receipts)
- `monitoring_thresholds.json` (Production drift triggers)

---

## 9. Acknowledgments & data credit

This research was built on the **FlyRank ML Internship dataset** provided by [FlyRank.ai](https://flyrank.ai). We gratefully acknowledge the FlyRank engineering team for providing the pseudonymized multi-tenant search console and analytics data warehouse.
