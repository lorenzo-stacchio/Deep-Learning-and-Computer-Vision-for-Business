import pandas as pd

df = pd.DataFrame(columns=["name"])

# df["name"] =["AM3", "FM", "AM2"]


# df = df[df["name"].apply(lambda x: not x.startswith("AM"))]

df["name"] =[3, 2, 4]
df = df[df["name"].apply(lambda x: not str(x).startswith("3"))]

# print(df[not df["name"].isin("AM")])

# print(df["name"] == "AM3")
print(df)