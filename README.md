# Price Optimization Model – Linear Demand & Revenue Maximization

**Tech Stack:**  
Python · Pandas · NumPy · Scikit-learn (LinearRegression) · Seaborn · Matplotlib

---

## 🎯 Project Overview

This project builds a **price optimization framework** using:

- A **linear demand model** per product  
- Brand & flavour as categorical drivers  
- Product attributes (volume, pack size)  
- Analytics-based optimal price calculation  

The key idea:

> Estimate how **units sold** change with **price**,  
> then compute the **revenue-maximizing price** for each product.

The project also includes **scenario analysis**, such as:

- What happens to total revenue if all prices are decreased by 10%?  
- What happens if we **remove low-performing brands** from the assortment?

---

## 📂 Data & Feature Setup

Two source tables:

- `product.csv`  
  - product_id  
  - brand  
  - flavour  
  - volume_per_joghurt_g  
  - packsize  

- `sales.csv`  
  - product_id  
  - price  
  - units  
  - date
 

Data is merged on `product_id` and a new variable is created:

```python
df["revenue"] = df["price"] * df["units"]


🧠 Methodology
1️⃣ Exploratory Data Analysis

General checks with a custom check_df() utility

Categorical summaries:

brand

flavour

Group-level stats:

brand_stats → average units sold per brand

flavor_stats → average units per flavour

volume_stats → performance by pack volume

pack_stats → performance by pack size

Correlation matrix on numerical variables:

price

units

volume_per_joghurt_g

packsize

This reveals which attributes correlate most with demand (units).

2️⃣ Demand Modeling (Linear Regression)

A linear regression model is fit on merged data:

X = [price, volume_per_joghurt_g, packsize] 
    + one-hot(brand) 
    + one-hot(flavour)

y = units


Using:

model = LinearRegression()
model.fit(X, y)
r2_full = model.score(X, y)


A baseline model without price is also trained to measure the incremental explanatory power of price:

X_base = [volume_per_joghurt_g, packsize] 
         + one-hot(brand) 
         + one-hot(flavour)
model_base = LinearRegression().fit(X_base, y)
r2_base = model_base.score(X_base, y)


This allows us to quantify:

How much of demand variation is explained by price vs. other attributes.

3️⃣ Product-Level Demand Spec

For each product i, the model implies a demand function:

units_i(price) ≈ a_i + b * price


Where:

b = global price coefficient (beta_price, same across products)

a_i = product-specific intercept (units at price=0, given its brand/flavour/attributes)

For each product, we compute a_i by predicting units with price = 0:

row0 = {price=0, volume, packsize, brand, flavour}
a_i = model.predict(X0)

4️⃣ Optimal Price Formula

Using microeconomics:

revenue_i(p) = p * units_i(p) 
             = p * (a_i + b * p)

=> optimal price p* = -a_i / (2b), if b < 0


Implementation details:

If beta_price < 0 (demand decreases with price), compute p* = -a_i / (2 * beta_price)

Clip p* into a realistic range based on observed prices:

pmin, pmed, pmax = min, median, max observed prices for that product
low  = max(0.1, 0.8 * pmin)
high = 1.2 * pmax
p*   = clip(p*, low, high)


If beta_price >= 0 (no negative slope), fall back to median price.

Then:

u_star = model.predict(at price = p*)
r_star = p* * u_star


All products’ results are collected into optimal_df:

Product ID

Brand

Flavor

Volume (g)

Pack Size

Optimal Price

Predicted Units @ p*

Predicted Revenue @ p*

Sorted by Pred Revenue @ p* to see the top revenue opportunities.

📊 Scenario Analysis
🔹 1) 10% Price Cut Scenario

Simulate a uniform -10% price cut across all products:

df_promo = df.copy()
df_promo['price'] = df_promo['price'] * 0.90
units_promo = model.predict(build_design_from_rows(df_promo[...]))
revenue_promo = (df_promo['price'] * units_promo).sum()


This gives a single total revenue number under a -10% price scenario,
which can be compared to the base revenue.

🔹 2) Removing Low-Performing Brands

Automatically detect brands in the bottom 10% of average units sold:

brand_perf = df.groupby("brand")["units"].mean().sort_values()
cutoff = brand_perf.quantile(0.10)
low_brands = brand_perf[brand_perf <= cutoff].index.tolist()
df_no_low = df[~df["brand"].isin(low_brands)]
revenue_no_low = df_no_low["revenue"].sum()


This lets us answer:

“What if we stopped selling low-performing brands?”

“Does total revenue go up or down if we prune the tail?”

💼 Business Value

This model enables:

Product-Level Pricing Decisions
Compute an optimal price per SKU based on demand sensitivity.

What-If Analysis
Simulate discount campaigns and product delisting.

Assortment Optimization
Quantify the revenue contribution of removing low-performing brands.

Structured Pricing Logic
Moves away from intuition-based pricing toward data-driven optimization.
