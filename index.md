# Insurance Marketing Mix Model

## Overview

This model estimates the relationship between marketing spend and policy conversions for Insure Co. across three acquisition channels: **Paid Search**, **Paid Social**, and **Email**. The goal is to determine optimal budget allocation to maximize total profit.

## Key Finding

**Email is the most profitable channel per marketing dollar spent, despite having the lowest profit per conversion.**

| Channel | Total Spend | Total Conversions | Avg Profit/Conv | ROI | Recommendation |
|---------|-------------|-------------------|-----------------|-----|----------------|
| Email | $60,713 | 449 | $2,318 | **17.1x** | Increase ~4x |
| Paid Social | $249,616 | 834 | $2,561 | 8.6x | Reduce |
| Paid Search | $614,901 | 1,298 | $4,727 | 10.0x | Trim |

The counterintuitive insight: email has the lowest profit per conversion ($2,318 vs $4,727 for search), yet generates the highest ROI because its acquisition cost ($8/lead) is dramatically lower than other channels.

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
- **K** = Maximum achievable conversions per month (saturation ceiling)
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
| Search | **0.57** | Clear spend→conversion relationship |
| Social | 0.44 | Moderate fit; noisier near saturation |
| Email | 0.50 | Good fit on the steep part of the curve |

### Fitted Parameters

| Channel | K (Max Conv/mo) | S (Half-Sat) | Saturation Proximity | Avg Profit/Conv |
|---------|-----------------|--------------|----------------------|-----------------|
| Email | 91 | $18,552 | 8% (far from saturation) | $2,318 |
| Social | 32 | $2,967 | 70% (near saturation) | $2,561 |
| Search | 45 | $7,637 | 69% (near saturation) | $4,727 |

### Optimal Budget Allocation

For a fixed monthly budget of $25,701:

| Channel | Current | Optimal | Change |
|---------|---------|---------|--------|
| Search | $17,081 | $14,167 | **-$2,914** |
| Social | $6,934 | $5,089 | **-$1,845** |
| Email | $1,686 | $6,445 | **+$4,759** |

At optimal, marginal profit per dollar is equalized across channels.

---

## Files

| File | Description |
|------|-------------|
| `MMM/insurance_marketing_mix_model.py` | Main model with ROI-saturation constraint |
| `notebooks/generate_insurance_data_v2.py` | Synthetic data generator (spend-driven response + cross-sell) |

### Running the Model

```bash
python notebooks/generate_insurance_data_v2.py   # regenerate the dataset
python MMM/insurance_marketing_mix_model.py       # fit the model + render charts
```

---

## Recommendations

1. **Increase email spend** from ~$1,686/month to ~$6,445/month (~4x increase)
2. **Trim search and social** (both near saturation) and reallocate to email
3. **Run controlled experiments** to validate predictions before large budget shifts
