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
