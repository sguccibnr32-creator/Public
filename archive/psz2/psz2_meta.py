import urllib.request, re, pandas as pd

BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# まずテーブル一覧を確認
url_meta = BASE + '?-source=J/A+A/594/A27&-out.max=1'
print('=== テーブル構造確認 ===')
data = urllib.request.urlopen(url_meta, timeout=30).read().decode('utf-8','ignore')
# FIELDタグから列名抽出
fields = re.findall(r'<FIELD[^>]+name="([^"]+)"', data)
print('列名:', fields)
print()
print(data[:3000])
