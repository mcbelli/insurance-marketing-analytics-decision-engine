---
---

# Exploratory Data Analysis

[← Back to Home](../../index.md)

---

This page presents eight exploratory analyses that reveal the key dynamics of insurance marketing. Each analysis surfaces insights that inform the marketing mix model and budget optimization.

---

## 1. Credit Score Impact on Conversion and Loss Ratio

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Credit-based insurance scores are a major underwriting tool. This analysis validates that better credit correlates with **both** higher conversion rates AND lower claims.

**Key Finding:** Excellent credit customers convert at ~2.4x the rate of Poor credit (11.8% vs 4.9%), with loss ratios 23 percentage points lower (0.39 vs 0.62).

**Implication:** Credit score should inform lead prioritization, not just pricing.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_credit_score.png">
  <img src="EDA_credit_score.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 2. Optimal Age Bands by Product

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Age is the primary rating variable in life and health insurance. The optimal customer age differs by product line.

**Key Finding:** 
- LTV peaks between ages 35 and 55, varying by product (Life earlier, Health and P&C later)
- Conversion rates show different age patterns across products
- Targeting should be product-specific

**Implication:** One-size-fits-all age targeting leaves value on the table.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_age_bands.png">
  <img src="EDA_age_bands.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 3. Cross-Sell and Multi-Product Opportunity

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Bundled customers retain roughly twice as long (~90% vs ~80% annual retention). This analysis quantifies the cross-sell opportunity.

**Key Finding:** Multi-product leads convert ~2x better (14.2% vs 6.8% for two-product vs single), and customers who actually bundle retain ~2x longer and carry ~2x the lifetime value ($6,096 vs $3,826).

**Implication:** Invest in cross-sell programs—bundling lifts both conversion and retention.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_cross_sell.png">
  <img src="EDA_cross_sell.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 4. Geographic Performance Variation

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Insurance is state-regulated—each state has different rate approval processes, coverage mandates, and competitive dynamics.

**Key Finding:** Loss ratios vary significantly by state, from ~15% to ~96%.

**Implication:** Geographic risk pricing and targeted underwriting are essential. The highest-loss states should be addressed immediately.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_geographic.png">
  <img src="EDA_geographic.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 5. Adverse Selection by Marketing Channel

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Cheaper acquisition channels are designed to attract marginally higher-risk customers. This analysis looks for that adverse-selection effect.

**Key Finding:** The effect is real but modest and partly masked by product mix. Holding product constant, the cheapest channel does carry more risk—within Health, email policies have a higher early-claim rate than search (29.7% vs 26.3%)—but at the aggregate level the channel signal is small relative to product-driven claims and sampling noise.

**Implication:** Any channel-level risk adjustment should be applied within product, not on raw channel averages.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_early_claims.png">
  <img src="EDA_early_claims.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 6. Marketing ROI by Channel

<table>
<tr>
<td width="50%" valign="top" markdown="1">

What matters for budget allocation is **profit per marketing dollar**, not profit per policy. A channel with lower profit per policy can still be better if acquisition costs are low enough.

**Key Finding:** Email earns the highest ROI (~17.9x, roughly 2x search and social) despite lower profit per policy, because its acquisition cost ($8/lead) is far below paid search ($45/lead).

**Implication:** Shift budget toward email to maximize total profit.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_policy_profitability.png">
  <img src="EDA_policy_profitability.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 7. State-Level Claims and Loss Ratios

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Identifying geographic risk concentration to inform pricing and underwriting decisions.

**Key Finding:** Loss ratios span ~15% to ~96% across states, concentrating risk in a handful of states that warrant rate increases or stricter underwriting.

**Implication:** State-level performance monitoring should be routine.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_state_claims.png">
  <img src="EDA_state_claims.png" width="100%">
</a>

</td>
</tr>
</table>

---

## 8. Bind Rate vs. Claims Rate Trade-off

<table>
<tr>
<td width="50%" valign="top" markdown="1">

Do higher-converting channels produce riskier policies? This tests the quality-quantity trade-off.

**Key Finding:** A positive correlation (r=0.67) across channel-product segments confirms that segments which bind aggressively also carry more claims—driven by Health, which both converts best and claims most.

**Implication:** Conversion optimization must be balanced against underwriting quality.

</td>
<td width="50%" valign="top" markdown="1">

<a href="EDA_bind_vs_claims.png">
  <img 