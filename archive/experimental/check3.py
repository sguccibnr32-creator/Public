import pandas as pd

PATH = r'D:\ドキュメント\エントロピー\新しいフォルダー (3)\931720.csv.gz.1'
df = pd.read_csv(PATH, comment='#', nrows=100)

print('=== 実際の列名（repr表示） ===')
for i, c in enumerate(df.columns):
    print(i, repr(c))

print()
print('=== 先頭3行 ===')
print(df.head(3).to_string())
