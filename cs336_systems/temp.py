import pandas as pd
df = pd.read_csv("data/results_fa_v_torch.csv", on_bad_lines='skip')
# df =
print(df.to_latex(index=False))

