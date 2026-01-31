# Marketing Mix Model

[← Back to Home](../index.md)

---

This page documents the Marketing Mix Model (MMM) that quantifies the relationship between marketing spend and customer conversions, enabling data-driven budget optimization.

---

## Model Overview

<table>
<tr>
<td width="55%" valign="top">

The model uses a **two-stage approach**:

1. **Response Curve**: Spend → Conversions (fit with Hill function)
2. **Profit Calculation**: Conversions × Avg Profit per Conversion

This separation produces tighter fits because the spend→conversions relationship is more direct than spend→profit (which includes claim variance).

**Model Fit (R²):**
| Channel | R² |
|---------|-----|
| Search | 0.26 |
| Social | 0.02 |
| Email | 0.05 |

</td>
<td width="45%" valign="top">

<a href="https://raw.githubusercontent.com/mcbelli/insurance-marketing-analytics-decision-engine/main/MMM/constrained_comparison.png">
  <img src="https://raw.githubusercontent.com/mcbelli/insurance-marketing-analytics-decision-engine/main/MMM/constrained_comparison.png" width="100%">
</a>
<em>Click to enlarge</em>

</td>
</tr>
</table>

---

## The Hill Saturation Function

Marketing spend exhibits **diminishing returns**—each additional dollar produces less than the last—and eventually **saturates**—there's a ceiling on what a channel can deliver.

The Hill function captures both:

```
Conversions(Spend) = K × Spend^β / (S^β + Spend^β)
```

**Parameters:**
- **K** = Maximum conversions per week (saturation ceiling)
- **S** = Half-saturation point (spend at which conversions = K/2)
- **β** = Shape parameter (steepness of the curve)

<a href="https://raw.githubusercontent.com/mcbelli/insurance-marketing-analytics-decision-engine/main/MMM/MMM_homepage.png">
  <img src="https://raw.githubusercontent.com/mcbelli/insurance-marketing-analytics-decision-engine/main/MMM/MMM_homepage.png" width="80%">
</a>

---

## Economic Assumptions

Profit is calculated as **NPV of policy cash flows**, not gross margin:

| Assumption | Value | Rationale |
|------------|-------|-----------|
| Expense ratio | 30% | Operating costs as % of premium |
| Discount rate | 10% | Time value of money |

```
Annual profit = Annual premium × (1 - 0.30) - Annual claims
NPV = Annual profit × Annuity factor(10%, tenure)
```

---

## ROI-Saturation Constraint

A key innovation: the model enforces that **higher-ROI channels are further from saturation**.

**Intuition:** If a channel has high average ROI, it's likely because we're operating on the steep part of the response curve (far from saturation). Low average ROI suggests we're already in diminishing returns.

**Constraint:**
```
If ROI_i > ROI_j, then (Spend_i / S_i) < (Spend_j / S_j)
```

This prevents the model from incorrectly concluding that high-ROI channels are "saturated" when we simply haven't tested higher spend levels.

---

## Fitted Parameters

| Channel | K (Max Conv/wk) | S (Half-Sat) | β | Sat. Proximity | Avg Profit/Conv |
|---------|-----------------|--------------|---|----------------|-----------------|
| Email | 26.4 | $2,509 | 0.90 | **0.08** (far) | $1,054 |
| Social | 11.6 | $2,861 | 0.59 | 0.47 | $3,059 |
| Search | 18.7 | $3,577 | 1.80 | 0.79 (near) | $2,202 |

**Interpretation:**
- **Email** is at just 8% of half-saturation—significant room to scale
- **Search** is at 79% of half-saturation—approaching diminishing returns
- **Social** is in between

---

## Optimal Budget Allocation

Given the fitted curves, the optimal allocation **equalizes marginal profit** across channels:

| Channel | Current | Optimal | Change |
|---------|---------|---------|--------|
| Search | $2,821/wk | $2,875/wk | +$54 |
| Social | $1,345/wk | $713/wk | **-$632** |
| Email | $199/wk | $778/wk | **+$579** |

The optimization shifts ~$600/week from social (near half-saturation) to email (far from saturation), while keeping search roughly constant.

---

## Model Limitations

1. **Limited spend variation**: Weekly spend varies only 2-12x within each channel. The model extrapolates saturation behavior beyond observed ranges.

2. **No carryover effects**: Assumes spending in week N only affects conversions in week N. Brand effects and delayed conversions are ignored.

3. **No interaction effects**: Channels are modeled independently. In reality, search and social may reinforce each other.

4. **Low R² for some channels**: Social and email have weak fits due to limited spend variation and small sample sizes.

---

## Files

| File | Description |
|------|-------------|
| `MMM/marketing_mix_model.py` | Main model code |
| `MMM/constrained_comparison.png` | Response curves visualization |
| `MMM/MMM_homepage.png` | Summary chart for homepage |
| `MMM/constrained_results.json` | Fitted parameters |
| `MMM/constrained_report.txt` | Full text report |

---

[← Back to Home](../index.md) | [Previous: EDA](../EDA/exploratory-analysis.md) | [Next: Optimization →](../Optimization/optimization.md)
