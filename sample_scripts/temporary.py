import glob
import pandas as pd
csv_files = glob.glob('*.csv')

for f in csv_files:
    df = pd.read_csv(f)
    d = list(df["SYMBOL"])
    print(f)
    print(d)
    print("\n")