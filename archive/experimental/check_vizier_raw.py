import urllib.request

# まずレスポンスの生データを確認
url = 'http://vizier.cds.unistra.fr/viz-bin/votable?-source=B/planck/plancksz2&-out=PSZ2&-out=RAJ2000&-out=DEJ2000&-out=Redshift&-out.max=5'
print('URL:', url)
try:
    data = urllib.request.urlopen(url, timeout=30).read().decode('utf-8','ignore')
    print('レスポンス長:', len(data), 'バイト')
    print()
    print('=== 先頭1000文字 ===')
    print(data[:1000])
    print()
    print('=== 末尾500文字 ===')
    print(data[-500:])
except Exception as e:
    print('エラー:', e)
