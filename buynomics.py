
##############################
# Mueller price optimisation
##############################

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

prod.head()
prod.shape
prod.info()
sal.head()
sal.shape()

# Load dataframes
prod = pd.read_csv("datasets/product.csv")
sal = pd.read_csv("datasets/sales.csv")

# Merge
df = sal.merge(prod, on="product_id", how="left")

# Compute revenue
df["revenue"] = df["price"] * df["units"]

# date parse
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# drop unnecessary columns
df = df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], errors="ignore")

##################################
# DISCOVERY DATA ANALYSIS
##################################

##################################
# GENERAL PICTURE
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)


##################################
# CAPTURE OF NUMERICAL AND CATEGORY VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



##################################
# ANALYSIS OF CATEGORY VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col,True)



##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# CORRELATION
##################################

# By brand
brand_stats = df.groupby('brand')['units'].agg(['mean', 'std', 'min', 'max', 'count']).round(1)
print("Brand stats:\n", brand_stats)

# By flavor
flavor_stats = df.groupby('flavour')['units'].agg(['mean', 'std', 'min', 'max', 'count']).round(1)
print("Flavor stats:\n", flavor_stats)

# By volume
volume_stats = df.groupby('volume_per_joghurt_g')['units'].agg(['mean', 'std', 'min', 'max', 'count']).round(1)
print("Volume stats:\n", volume_stats)

# By packsize
pack_stats = df.groupby('packsize')['units'].agg(['mean', 'std', 'min', 'max', 'count']).round(1)
print("Pack stats:\n", pack_stats)


df[num_cols].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


##################################
# MODEL
##################################
#Dummies
brand_dummies  = pd.get_dummies(df['brand'],   prefix='brand')   if 'brand'   in df.columns else pd.DataFrame(index=df.index)
flavor_dummies = pd.get_dummies(df['flavour'], prefix='flavor')  if 'flavour' in df.columns else pd.DataFrame(index=df.index)


num_cols = [c for c in ['price', 'volume_per_joghurt_g', 'packsize'] if c in df.columns]
X = pd.concat([df[num_cols], brand_dummies, flavor_dummies], axis=1).fillna(0)
y = df['units'].values


model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("R2 full:", model.score(X, y))

# Baseline (except price)
base_cols = [c for c in ['volume_per_joghurt_g', 'packsize'] if c in df.columns]
X_base = pd.concat([df[base_cols], brand_dummies, flavor_dummies], axis=1).fillna(0)
model_base = LinearRegression().fit(X_base, y) if X_base.shape[1] > 0 else None
r2_base = model_base.score(X_base, y) if model_base is not None else np.nan
print("R2 base (no price):", r2_base)

##################################
#  Helper: Column Aligner
##################################
feature_cols = X.columns.tolist()

def build_design_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
"""
Builds a design matrix aligned with training features (X).
Input rows may contain a subset of: price, volume_per_joghurt_g, packsize, brand, flavour.
Returns a DataFrame with the same columns as X, filling missing ones with 0
"""
    parts = []
# numeric features
    if 'price' in rows.columns:
        parts.append(rows[['price']])
    for c in ['volume_per_joghurt_g', 'packsize']:
        if c in rows.columns:
            parts.append(rows[[c]])
# categorical dummies
    if 'brand' in rows.columns:
        parts.append(pd.get_dummies(rows['brand'], prefix='brand'))
    if 'flavour' in rows.columns:
        parts.append(pd.get_dummies(rows['flavour'], prefix='flavor'))
    Xi = pd.concat(parts, axis=1).fillna(0) if parts else pd.DataFrame(index=rows.index)
# align with training columns
    for col in feature_cols:
        if col not in Xi.columns:
            Xi[col] = 0
    Xi = Xi[feature_cols]
    return Xi


##################################
#  Optimization (Linear Spec)
##################################
""" Model logic:
   units ≈ a_i + b * price
  revenue(p) = p * (a_i + b * p)
   Optimal price (p*) = -a_i / (2b) 
"""

#1)Product list
unique_products = (
    prod.set_index('product_id')
    if 'product_id' in prod.columns
    else df[['product_id', 'brand', 'flavour', 'volume_per_joghurt_g', 'packsize']]
        .drop_duplicates()
        .set_index('product_id')
)

#2)Extract price coefficient
beta_price = float(
    model.coef_[feature_cols.index('price')]
    if 'price' in feature_cols
    else np.nan
)

#3)Observed price bounds (used for clipping unrealistic p*)
price_bounds = (
    df.groupby('product_id')['price']
    .agg(['min', 'median', 'max'])
    .rename(columns={'min': 'pmin', 'median': 'pmed', 'max': 'pmax'})
)

optimal_rows = []

for pid in unique_products.index:
# Product attributes
    pr = unique_products.loc[pid]
    pmin, pmed, pmax = price_bounds.loc[pid, ['pmin', 'pmed', 'pmax']]

# Predicted units at price=0 -> a_i
    row0 = pd.DataFrame([{
        'price': 0.0,
        'volume_per_joghurt_g': pr.get('volume_per_joghurt_g', np.nan),
        'packsize': pr.get('packsize', np.nan),
        'brand': pr.get('brand', None),
        'flavour': pr.get('flavour', None),
    }])
    X0 = build_design_from_rows(row0)
    a_i = float(model.predict(X0)[0])

# Compute optimal price
    if beta_price < 0:
        p_star = -a_i / (2.0 * beta_price)
# Clip to realistic range (within ±20% of observed)
        low, high = max(0.1, 0.8 * pmin), 1.2 * pmax
        p_star = float(np.clip(p_star, low, high))
    else:
# If the coefficient is non-negative, revert to median price
        p_star = float(pmed)

# Predicted units & revenue at p*
    row_star = row0.copy()
    row_star['price'] = p_star
    X_star = build_design_from_rows(row_star)
    u_star = float(model.predict(X_star)[0])
    r_star = max(0.0, p_star) * max(0.0, u_star)

    optimal_rows.append({
        'Product ID': pid,
        'Brand': pr.get('brand', None),
        'Flavor': pr.get('flavour', None),
        'Volume (g)': pr.get('volume_per_joghurt_g', None),
        'Pack Size': pr.get('packsize', None),
        'Beta_price': round(beta_price, 6),
        'Base units @ p=0': round(a_i, 2),
        'Optimal Price': round(p_star, 2),
        'Pred Units @ p*': round(u_star, 1),
        'Pred Revenue @ p*': round(r_star, 2),
    })

optimal_df = (
    pd.DataFrame(optimal_rows)
    .sort_values('Pred Revenue @ p*', ascending=False)
)
print("Optimal Price Table:\n", optimal_df.head(10))


##################################
#  Promotion Scenario (−10%)
##################################
df_promo = df.copy()
df_promo['price'] = df_promo['price'] * 0.90

X_promo = build_design_from_rows(
    df_promo[['price', 'volume_per_joghurt_g', 'packsize', 'brand', 'flavour']]
)
units_promo = model.predict(X_promo)
revenue_promo = float((df_promo['price'] * units_promo).sum())
print("Promo revenue (−10% price cut):", round(revenue_promo, 2))


##################################
#  “Low Performer Brands” Scenario (Dynamic)
##################################
# Automatically detect underperforming brands based on average units

brand_perf = df.groupby("brand")["units"].mean().sort_values()
cutoff = brand_perf.quantile(0.10)  # bottom 10%
low_brands = brand_perf[brand_perf <= cutoff].index.tolist()

df_no_low = df[~df["brand"].isin(low_brands)]
revenue_no_low = float(df_no_low["revenue"].sum())

print(f"Low-performing brands ({len(low_brands)}): {low_brands}")
print(f"Revenue excluding low-performing brands: {revenue_no_low:,.2f}")
