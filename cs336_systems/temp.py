import pandas as pd
df = pd.read_csv("data/2_1_3/results_2_1_3_1warmup.csv", on_bad_lines='skip')
print(df.to_latex())

