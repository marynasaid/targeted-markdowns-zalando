# Which products still need a discount? Targeted markdowns on Zalando data

**Question.** A fashion retailer discounts many products, often by 50%. Removing all discounts protects margin but loses sales. Which products still need a discount, and how deep?

**Data.** ACIC 2023 data challenge, prepared by Zalando SE (MIT licence): 3,908 products, 95 weeks of sales, weekly discounts of 0–50%. For five test weeks the data contains sales at **every** discount level, so the profit of any pricing choice can be measured exactly. Put `training_sample.csv` and `testing_sample.csv` next to the scripts.

## Approach
- Each product's sales without a discount form its baseline; the effect of each discount level is estimated by product segment with Monte Carlo draws from that baseline.
- The most profitable discount is chosen per product (gross margin 60% at full price assumed).
- Compared with current pricing, a flat 20% discount, no discounts, a gradient-boosting forecast model, and the best possible choice if true demand were known.

## Results (gross profit over five test weeks, vs current pricing)

| Strategy | Products discounted | Profit change |
|---|---|---|
| 20% discount for everyone | 100% | −9.2% |
| No discounts | 0% | +7.7% |
| Gradient boosting, discount per product | 73% | +8.5% |
| **My method, discount per product** | **40% (all at 20%)** | **+10.6%** |
| Best possible | — | +15.2% |

- Captures 70% of the best possible gain.
- The recommendation reduces to a simple rule: three product categories keep a 20% discount, three return to full price, no product needs 50%.
- **Stability:** five reruns with different random draws gave +10.6% to +10.7%; across 2,000 bootstrap resamples of products the method was ahead of current pricing, of no discounts and of the ML model in 100% of resamples.

## Limitations
- One dataset and five test weeks; sales are simulated.
- Depends on margin: at 50% margin +21.4% (best), at 70% margin +3.2% (the ML model is slightly ahead at +3.9%).
- Discount levels are fixed (0–50% in steps of 10); intermediate levels cannot be tested.

## How this was built
Method design, research questions and validation choices are my own. I used an AI assistant (Claude) to help refactor the code.

## How to run
```
pip install -r requirements.txt
python forecast_and_effects.py   # effect accuracy vs forecasting models
python discount_policy.py        # profit of each pricing strategy
python stability_check.py 1      # rerun with random seed 1
```

