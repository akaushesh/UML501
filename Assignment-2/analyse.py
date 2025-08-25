import pandas as pd
import numpy as np
import math

# --- Configuration ---
csv_file = 'AWCustomers.csv'     # Path to your CSV file
attribute = 'Gender'              # Column name to analyze
range_size = 10000000                  # Bin width (can be float)
# ----------------------

# Load CSV
df = pd.read_csv(csv_file)
column = df[attribute]

# --- Basic Stats ---
missing_count = column.isnull().sum()
empty_string_count = (column == '').sum() if column.dtype == object else 0

numeric_count = 0
numeric_string_count = 0
non_numeric_string_count = 0

# Classify values
for val in column.dropna():
    val_str = str(val).strip()
    if val_str == '':
        continue  # Already counted as empty string
    try:
        float_val = float(val_str)
        if isinstance(val, (int, float)):
            numeric_count += 1
        else:
            numeric_string_count += 1
    except ValueError:
        non_numeric_string_count += 1

# --- Output type classification summary ---
print(f"\n🧾 Detailed Type Analysis for column '{attribute}':")
print(f"❓ Missing (NaN/None): {missing_count}")
print(f"🔲 Empty strings: {empty_string_count}")
print(f"🔢 True numeric values: {numeric_count}")
print(f"🔢 Numeric strings: {numeric_string_count}")
print(f"🔤 Non-numeric strings: {non_numeric_string_count}")

# --- Convert all to numeric for distribution ---
converted = pd.to_numeric(column, errors='coerce').dropna()

# --- Check for binary data ---
unique_vals = sorted(converted.unique())
if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
    print(f"\n📊 Binary Frequency for '{attribute}':")
    print(f"0: {(converted == 0).sum()}")
    print(f"1: {(converted == 1).sum()}")

elif converted.empty:
    print("\n⚠️ No valid numeric data to analyze for distribution.")

else:
    # --- Numeric range and binning ---
    actual_min = converted.min()
    actual_max = converted.max()
    print(f"\n✅ Numeric Range: {actual_min} to {actual_max}")

    # Bin start/end aligned with range_size (supports decimal)
    min_val = math.floor(actual_min / range_size) * range_size
    max_val = math.ceil(actual_max / range_size) * range_size
    bins = np.arange(min_val, max_val + range_size, range_size)

    # Create readable bin labels
    labels = [f"{round(bins[i], 2)}–{round(bins[i+1], 2)}" for i in range(len(bins) - 1)]

    # Bin the values
    binned = pd.cut(converted, bins=bins, labels=labels, right=False)
    counts = binned.value_counts().sort_index()

    print(f"\n📊 Frequency Distribution (bin size = {range_size}):")
    for label, count in counts.items():
        print(f"{label}: {count}")

# --- Unique value analysis (all types) ---
non_null_values = column.dropna().astype(str).str.strip()
unique_values = non_null_values[non_null_values != ''].unique()
unique_count = len(unique_values)

print(f"\n🔢 Total unique values in '{attribute}' (excluding NaN/None): {unique_count}")

# Optional: display top 10 unique values
# print("\n🔍 Sample unique values:")
# for val in unique_values[:10]:
#     print(f"• {repr(val)}")


