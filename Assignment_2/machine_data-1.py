import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

# 1) Read data
df = pd.read_csv("machine_data-1.csv")

# 2) Clean up
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["manufacturef"] = df["manufacturef"].astype(str).str.upper()

print("\nHEAD:\n", df.head())
print("\nCOLUMNS:\n", df.columns)

# 3) Q1 + Q2: range + expected load/time per manufacturer
summary = df.groupby("manufacturef").agg(
    load_min=("load", "min"),
    load_max=("load", "max"),
    time_min=("time", "min"),
    time_max=("time", "max"),
    load_mean=("load", "mean"),
    load_median=("load", "median"),
    time_mean=("time", "mean"),
    time_median=("time", "median"),
).round(3)

print("\nSUMMARY (per manufacturer):\n", summary)

summary.to_csv("summary_table.csv", index=True)  # optional

# Plot ranges visually (boxplots)
plt.figure()
df.boxplot(column="load", by="manufacturef")
plt.title("Load by Manufacturer")
plt.suptitle("")
plt.xlabel("Manufacturer")
plt.ylabel("Load")
plt.tight_layout()
plt.savefig("plot_load_boxplot.png", dpi=200)
plt.show()

plt.figure()
df.boxplot(column="time", by="manufacturef")
plt.title("Time by Manufacturer")
plt.suptitle("")
plt.xlabel("Manufacturer")
plt.ylabel("Time")
plt.tight_layout()
plt.savefig("plot_time_boxplot.png", dpi=200)
plt.show()

# 4) Q3: relationship load vs time
plt.figure()
for m, g in df.groupby("manufacturef"):
    r = g["load"].corr(g["time"])
    print(f"{m}: corr(load,time) = {r:.3f}")
    plt.scatter(g["load"], g["time"], s=12, label=f"{m} (r={r:.2f})")

plt.title("Load vs Time (scatter)")
plt.xlabel("Load")
plt.ylabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("plot_load_vs_time.png", dpi=200)
plt.show()

# 5) Q4 + Q5: best distribution by AIC (load & time)
def aic(dist, data):
    params = dist.fit(data)
    ll = np.sum(dist.logpdf(data, *params))
    k = len(params)
    return 2*k - 2*ll, params

dists = {
    "Normal": st.norm,
    "Lognormal": st.lognorm,
    "Exponential": st.expon,
    "Weibull": st.weibull_min
}

def best_fit(colname):
    data = df[colname].values
    results = []
    for name, dist in dists.items():
        score, params = aic(dist, data)
        results.append((score, name, dist, params))
    results.sort(key=lambda x: x[0])
    return results[0], results

for col in ["load", "time"]:
    best, allres = best_fit(col)
    print(f"\nBest distribution for {col}: {best[1]} (AIC={best[0]:.2f})")

    # histogram + fitted curve for best dist
    _, _, dist, params = best
    x = np.linspace(df[col].min(), df[col].max(), 300)
    y = dist.pdf(x, *params)

    plt.figure()
    plt.hist(df[col], bins=20, density=True)
    plt.plot(x, y)
    plt.title(f"{col} distribution fit: {best[1]}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"plot_{col}_distfit_{best[1].lower()}.png", dpi=200)
    plt.show()

print("\nDONE. Plots saved as .png files in this folder.")
