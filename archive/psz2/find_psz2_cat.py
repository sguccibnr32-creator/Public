import urllib.request, re

BASE = 'http://vizier.cds.unistra.fr/viz-bin/votable'

# カタログ名を検索
url = BASE + '?-source=B/planck&-out.max=1&-meta'
print('=== B/planck配下のカタログ確認 ===')
try:
    data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
    print(data[:2000])
except Exception as e:
    print('エラー:', e)

print()

# PSZ2論文のVizierカタログはJ/A+A/594/A27
url2 = BASE + '?-source=J/A+A/594/A27&-out.max=1&-meta'
print('=== J/A+A/594/A27 確認 ===')
try:
    data2 = urllib.request.urlopen(url2, timeout=30).read().decode('utf-8','ignore')
    print(data2[:2000])
except Exception as e:
    print('エラー:', e)
