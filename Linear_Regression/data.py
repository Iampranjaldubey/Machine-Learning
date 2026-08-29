import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Generate CGPA between 5.0 and 9.9
cgpa = np.round(np.random.uniform(5.0, 9.9, n), 2)

# Generate package with correlation + noise
package = 1.5 * cgpa - 4 + np.random.normal(0, 0.8, n)
package = np.round(np.clip(package, 2.0, None), 2)  # minimum package 2 LPA

df = pd.DataFrame({
    "CGPA": cgpa,
    "Package_LPA": package
})

file_path = "cgpa_package_1000.csv"
df.to_csv(file_path, index=False)

file_path
