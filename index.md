# Insurance Marketing Mix Model

## Overview

This model estimates the relationship between marketing spend and policy conversions for Insure Co. across three acquisition channels: **Paid Search**, **Paid Social**, and **Email**. The goal is to determine optimal budget allocation to maximize total profit.

## Key Finding

**Email is the most profitable channel per marketing dollar spent, despite having the lowest profit per conversion.**

| Channel | Total Spend | Total Conversions | Avg Profit/Conv | ROI | Recommendation |
|---------|-------------|-------------------|-----------------|-----|----------------|
| Email | $31,504 | 321 | $1,054 | **10.7x** | Increase 4x |
| Paid Social | $212,555 | 692 | $3,059 | 10.0x | Reduce ~50% |
| Paid Search | $445,758 | 1,149 | $2,202 | 5.7x | Hold steady |

The counterintuitive insight: email has the lowest profit per conversion ($1,054 vs $3,059 for social), yet generates the highest ROI because its acquisition cost ($8/lead) is dramatically lower than other channels.

---

## Economic Assumptions

Profit is calculated as **NPV of policy cash flows**, not gross margin:

| Assumption | Value | Rationale |
|------------|-------|-----------|
| **Expense ratio** | 30% | Operating costs (admin, servicing, overhead) as % of premium |
| **Discount rate** | 10% | Annual rate for time value of money |

**Calculation:**
```
Annual profit = Annual premium × (1 - 0.30) - Annual claims
NPV = Annual profit × Annuity factor(10%, tenure)
```

---

## Model Specification

### Two-Stage Approach

The model separates marketing efficiency from unit economics:

1. **Response curve**: Spend → Conversions (fitted with Hill function)
2. **Profit calculation**: Conversions × Avg Profit per Conversion

This separation produces much tighter fits because the spend→conversions relationship is more direct than spend→profit.

### Functional Form

```
Conversions(Spend) = K × Spend^β / (S^β + Spend^β)
```

Where:
- **K** = Maximum achievable conversions per week (saturation ceiling)
- **S** = Half-saturation point (spend level at which conversions reach 50% of K)
- **β** = Shape parameter (controls steepness of the curve)

### ROI-Saturation Constraint

The model enforces that higher-ROI channels are further from saturation:

```
If ROI_i > ROI_j, then (Current_Spend_i / S_i) < (Current_Spend_j / S_j)
```

This prevents the model from concluding that high-ROI channels are "saturated" when in reality we just haven't tested higher spend levels.

---

## Results

### Model Fit (R²)

| Channel | R² (conversions) | Interpretation |
|---------|------------------|----------------|
| Search | **0.26** | Good fit - clear spend→conversion relationship |
| Social | 0.02 | Weak fit - high variance in conversions |
| Email | 0.05 | Weak fit - limited spend variation |

### Fitted Parameters

| Channel | K (Max Conv/wk) | S (Half-Sat) | Saturation Proximity | Avg Profit/Conv |
|---------|-----------------|--------------|----------------------|-----------------|
| Email | 26.4 | $2,509 | 0.08 (far from saturation) | $1,054 |
| Social | 11.6 | $2,861 | 0.47 (below half-sat) | $3,059 |
| Search | 18.7 | $3,577 | 0.79 (near half-sat) | $2,202 |

### Optimal Budget Allocation

For a fixed weekly budget of $4,366:

| Channel | Current | Optimal | Change |
|---------|---------|---------|--------|
| Search | $2,821 | $2,875 | +$54 |
| Social | $1,345 | $713 | **-$632** |
| Email | $199 | $778 | **+$579** |

At optimal, marginal profit per dollar is equalized across channels.

---

## Files

| File | Description |
|------|-------------|
| `insurance_model_with_rules.py` | Main model with ROI-saturation constraint |
| `generate_insurance_data.py` | Synthetic data generator |

### Running the Model

```bash
python insurance_model_with_rules.py
```

Outputs saved to `./constrained_model_outputs/`

---

## Recommendations

1. **Increase email spend** from ~$200/week to ~$800/week (4x increase)
2. **Reduce social spend** by ~50% and reallocate to email
3. **Run controlled experiments** to validate predictions before large budget shifts
