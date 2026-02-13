# Import required libraries
import pandas as pd              # For reading and manipulating data tables
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from scipy import stats as st    # For probability distributions and fitting

# 1) Read data from CSV file into a pandas DataFrame
df = pd.read_csv("machine_data-1.csv")

# 2) Clean up

# If an unnecessary index column exists, remove it
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure manufacturer labels are strings and convert them to uppercase (A, B, C)
df["manufacturef"] = df["manufacturef"].astype(str).str.upper()

# Print first few rows of data to check structure
print("\nHEAD:\n", df.head())

# Print column names to verify dataset structure
print("\nCOLUMNS:\n", df.columns)

# 3) Q1 + Q2: Calculate range and expected values per manufacturer

# Group data by manufacturer and compute statistical summary
summary = df.groupby("manufacturef").agg(
    load_min=("load", "min"),         # Minimum load
    load_max=("load", "max"),         # Maximum load
    time_min=("time", "min"),         # Minimum time
    time_max=("time", "max"),         # Maximum time
    load_mean=("load", "mean"),       # Average load
    load_median=("load", "median"),   # Median load
    time_mean=("time", "mean"),       # Average time
    time_median=("time", "median"),   # Median time
).round(3)                            # Round values to 3 decimals

# Print summary table
print("\nSUMMARY (per manufacturer):\n", summary)

# Save summary table to CSV file (optional for report use)
summary.to_csv("summary_table.csv", index=True)

# ------------------------------
# Create boxplot for load
# ------------------------------

plt.figure()  # Create new figure window
df.boxplot(column="load", by="manufacturef")  # Boxplot of load grouped by manufacturer
plt.title("Load by Manufacturer")
plt.suptitle("")  # Remove automatic subtitle
plt.xlabel("Manufacturer")
plt.ylabel("Load")
plt.tight_layout()  # Adjust layout to avoid overlap
plt.savefig("plot_load_boxplot.png", dpi=200)  # Save image
plt.show()  # Display plot

# ------------------------------
# Create boxplot for time
# ------------------------------

plt.figure()
df.boxplot(column="time", by="manufacturef")
plt.title("Time by Manufacturer")
plt.suptitle("")
plt.xlabel("Manufacturer")
plt.ylabel("Time")
plt.tight_layout()
plt.savefig("plot_time_boxplot.png", dpi=200)
plt.show()

# 4) Q3: Relationship between load and time

plt.figure()

# Loop through each manufacturer group
for m, g in df.groupby("manufacturef"):

    # Compute correlation between load and time
    r = g["load"].corr(g["time"])

    # Print correlation value
    print(f"{m}: corr(load,time) = {r:.3f}")

    # Create scatter plot for this manufacturer
    plt.scatter(g["load"], g["time"], s=12, label=f"{m} (r={r:.2f})")

plt.title("Load vs Time (scatter)")
plt.xlabel("Load")
plt.ylabel("Time")
plt.legend()  # Show manufacturer labels
plt.tight_layout()
plt.savefig("plot_load_vs_time.png", dpi=200)
plt.show()

# 5) Q4 + Q5: Find best distribution using AIC

# Function to compute AIC value
def aic(dist, data):
    params = dist.fit(data)  # Fit distribution parameters to data
    ll = np.sum(dist.logpdf(data, *params))  # Log-likelihood
    k = len(params)  # Number of parameters
    return 2*k - 2*ll, params  # AIC formula

# Dictionary of candidate distributions
dists = {
    "Normal": st.norm,
    "Lognormal": st.lognorm,
    "Exponential": st.expon,
    "Weibull": st.weibull_min
}

# Function to find best-fitting distribution for a column
def best_fit(colname):
    data = df[colname].values  # Extract column values
    results = []

    # Try each distribution
    for name, dist in dists.items():
        score, params = aic(dist, data)
        results.append((score, name, dist, params))

    # Sort by lowest AIC (best fit)
    results.sort(key=lambda x: x[0])

    return results[0], results

# Apply distribution fitting for both load and time
for col in ["load", "time"]:

    best, allres = best_fit(col)

    print(f"\nBest distribution for {col}: {best[1]} (AIC={best[0]:.2f})")

    # Extract best distribution
    _, _, dist, params = best

    # Create smooth x values
    x = np.linspace(df[col].min(), df[col].max(), 300)

    # Compute probability density function
    y = dist.pdf(x, *params)

    # Plot histogram and fitted curve
    plt.figure()
    plt.hist(df[col], bins=20, density=True)
    plt.plot(x, y)
    plt.title(f"{col} distribution fit: {best[1]}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"plot_{col}_distfit_{best[1].lower()}.png", dpi=200)
    plt.show()

# Final confirmation message
print("\nDONE. Plots saved as .png files in this folder.")
