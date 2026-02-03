# Budget Optimization

[← Back to Home](../../mcbelli.github.io/index.md)

---

This page documents the budget optimization simulation that compares current allocation (Period 1) to optimal allocation (Period 2).

---

## Optimization Approach

<table>
<tr>
<td width="55%" valign="top">

Using the fitted response curves from the Marketing Mix Model, we simulate two scenarios over a 52-week period:

**Period 1 (Current):** Historical budget allocation
- Search: $2,821/week
- Social: $1,345/week  
- Email: $199/week
- **Total: $4,366/week**

**Period 2 (Optimal):** Budget reallocated to equalize marginal ROI
- Search: $2,875/week (+$54)
- Social: $713/week (-$632)
- Email: $778/week (+$579)
- **Total: $4,366/week** (same)

</td>
<td width="45%" valign="top">

<a href="optimization_comparison.png">
  <img src="optimization_comparison.png" width="100%">
</a>
<em>Click to enlarge</em>

</td>
</tr>
</table>

---

## Simulation Results

<a href="Optimization_homepage.png">
  <img src="Optimization_homepage.png" width="70%">
</a>

| Metric | Period 1 (Current) | Period 2 (Optimal) | Change |
|--------|-------------------|-------------------|--------|
| Marketing Spend | $227,029 | $227,029 | $0 |
| Conversions | 746 | 931 | **+185 (+24.8%)** |
| Profit | $1,699,462 | $1,801,738 | **+$102,277 (+6.0%)** |
| ROI | 7.5x | 7.9x | +0.5x |

---

## Why Does Reallocation Work?

The optimization works because **marginal ROI varies across channels**:

| Channel | Current Marginal ROI | At $500 Spend | At $2,000 Spend |
|---------|---------------------|---------------|-----------------|
| Email | High (steep curve) | ~$7.70 | ~$3.10 |
| Social | Medium | ~$8.10 | ~$2.60 |
| Search | Lower (flatter curve) | ~$4.10 | ~$7.10 |

At current allocations:
- **Email** ($199/wk) is on the steep part of its curve—high marginal returns
- **Social** ($1,345/wk) is past its half-saturation—diminishing returns
- **Search** ($2,821/wk) is near half-saturation—moderate returns

Moving dollars from social to email captures more conversions per dollar.

---

## Channel-Level Impact

### Email (Increased from $199 to $778/week)

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| Annual Spend | $10,348 | $40,456 | +$30,108 |
| Conversions | 127 | 391 | +264 |
| Profit | $133,953 | $412,214 | +$278,261 |
| ROI | 12.9x | 10.2x | -2.7x |

Email's ROI *decreases* as we spend more (diminishing returns), but it's still the highest-ROI channel at the new level.

### Social (Decreased from $1,345 to $713/week)

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| Annual Spend | $69,940 | $37,076 | -$32,864 |
| Conversions | 298 | 184 | -114 |
| Profit | $910,989 | $563,164 | -$347,825 |
| ROI | 13.0x | 15.2x | +2.2x |

Social's ROI *increases* as we spend less (moving back up the curve), but we're reallocating those dollars to a higher-opportunity channel.

### Search (Roughly Flat)

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| Annual Spend | $146,692 | $149,500 | +$2,808 |
| Conversions | 321 | 356 | +35 |
| Profit | $707,520 | $826,360 | +$118,840 |
| ROI | 4.8x | 5.5x | +0.7x |

---

## Optimization Theory

The optimal allocation satisfies the **equi-marginal principle**: 

> At optimum, the marginal profit per dollar is equal across all channels.

If marginal ROI were higher for one channel, we could improve total profit by shifting a dollar from a lower-marginal-ROI channel to it.

At the optimal allocation:
- Marginal profit (email) ≈ Marginal profit (social) ≈ Marginal profit (search) ≈ $6/dollar

---

## Caveats

1. **Model uncertainty**: The response curves have estimation error, especially for channels with limited spend variation.

2. **Extrapolation risk**: Email is currently at very low spend levels. The model extrapolates its performance at 4x higher spend.

3. **Execution factors**: Can email volume actually scale 4x? Are there list fatigue effects?

4. **Competitive response**: Competitors may respond to our channel shifts.

**Recommendation:** Implement the reallocation gradually and monitor for saturation signals (declining conversion rates, rising CPL).

---

## Files

| File | Description |
|------|-------------|
| `Optimization/optimize_budget.py` | Simulation code |
| `Optimization/optimization_comparison.png` | Channel-level comparison |
| `Optimization/Optimization_homepage.png` | Summary chart |
| `Optimization/optimization_results.json` | Simulation results |

---

[← Back to Home](../../mcbelli.github.io/index.md) | [Previous: MMM](../MMM/insurance_marketing-mix-model.md) | [Next: Business Impact →](../../mcbelli.github.io/index.md#4-business-impact)
