# GitHub v4.7.8 公開手順

## 前提
- 既存リポジトリ: `sguccibnr32-creator/Public`
- 最新タグ: `v4.7.6`
- 今回追加: `v4.7.8` (`v4.7.7` は内部バージョンのためスキップ)
- 配置場所: このzipを展開した `v478_release/` ディレクトリ

---

## ステップ 0: ローカルクローン (未クローンの場合のみ)

Windows PowerShell で:

```powershell
cd D:\
git clone https://github.com/sguccibnr32-creator/Public.git
cd Public
```

既にクローン済みの場合:

```powershell
cd D:\path\to\Public
git pull origin main
git status  # クリーンな状態であることを確認
```

---

## ステップ 1: v4.7.8 ブランチ作成

```powershell
# v4.7.6 の最新コミットから v4.7.8 ブランチを切る
git checkout main
git pull origin main
git checkout -b release/v4.7.8
```

---

## ステップ 2: ディレクトリ `v478_release/` をコピー

この zip を展開して得た `v478_release/` ディレクトリをリポジトリ直下にコピー:

```powershell
# zip 展開後の場所 (例: C:\Users\忍\Downloads\v478_release)
# リポジトリ直下に置く
xcopy /E /I /Y C:\Users\忍\Downloads\v478_release D:\path\to\Public\v478_release

# 確認
cd D:\path\to\Public
dir v478_release
```

想定される構造:
```
Public/
├── (既存の v4.7.6 ファイル群)
└── v478_release/
    ├── README.md
    ├── CHANGELOG.md
    ├── RELEASE_NOTES_v4.7.8.md
    ├── LICENSE
    ├── .gitignore
    ├── scripts/  (10 .py + README.md)
    ├── data/     (14 .csv/.json + README.md)
    ├── figures/  (10 .pdf/.png + README.md)
    ├── pdf/      (6 .pdf)
    └── docs/     (.html + .txt)
```

---

## ステップ 3: git add とコミット

```powershell
# 追加されるファイルを確認
git add v478_release/
git status

# 期待される追加: 約 50 ファイル
# 既存ファイルに影響が出ていないことを確認

# コミット
git commit -m "Add v4.7.8 release: Dwarf Spheroidal Extension

- Bernoulli prediction G_Strigari = s_0(1-s_0)*a_0 = 0.228 a_0
  verified in dSph (5%) and SPARC bridge outer (4%)
- J3 regime inversion confirmed in 28/31 dSph
- Continuous C15 -> Strigari transition in 4 bridge galaxies (all A-grade)
- g_obs M_bar-independence confirmed (0.2sigma null)
- Strict Strigari universality partially retracted (-3.4sigma r_h dependence)
- arXiv paper extended to 12 pages with new Section 7
- WordPress HTML updated with new Section 10"
```

---

## ステップ 4: GitHub へプッシュ

```powershell
git push -u origin release/v4.7.8
```

---

## ステップ 5: Pull Request を作成 (Web UI)

1. https://github.com/sguccibnr32-creator/Public を開く
2. 「Compare & pull request」ボタンが出るのでクリック
3. base: `main` ← compare: `release/v4.7.8`
4. Title: `Release v4.7.8 — Dwarf Spheroidal Extension`
5. Description には `RELEASE_NOTES_v4.7.8.md` の TL;DR セクションを貼る
6. "Merge pull request" で main にマージ

コマンドラインで直接マージする場合:

```powershell
git checkout main
git merge release/v4.7.8
git push origin main
```

---

## ステップ 6: タグ作成とリリース作成

### タグ作成 (コマンドライン)

```powershell
git checkout main
git pull origin main
git tag -a v4.7.8 -m "v4.7.8: Dwarf Spheroidal Extension

Pressure-supported dSph extension with Bernoulli prediction and
continuous C15->Strigari regime transition verification.

See v478_release/RELEASE_NOTES_v4.7.8.md for details."

git push origin v4.7.8
```

### GitHub Release 作成 (Web UI)

1. https://github.com/sguccibnr32-creator/Public/releases へ移動
2. 「Draft a new release」をクリック
3. 「Choose a tag」で `v4.7.8` を選択
4. Release title: `v4.7.8 — Dwarf Spheroidal Extension`
5. Description 欄に `v478_release/RELEASE_NOTES_v4.7.8.md` の内容を全文コピペ
   - または「Generate release notes」でコミットから自動生成したものを編集
6. **アセット添付** (推奨):
   - `v478_release/pdf/membrane_arxiv_v478.pdf` (投稿用原稿)
   - 必要なら `v478_release/` 全体を zip 化 (約 5-6 MB) して添付

   PowerShell で zip 化:
   ```powershell
   cd D:\path\to\Public
   Compress-Archive -Path v478_release -DestinationPath v478_release.zip
   ```
   `v478_release.zip` を Release assets にドラッグ&ドロップ

7. **Set as the latest release** にチェック
8. 「Publish release」をクリック

---

## ステップ 7: READMEトップページの更新 (オプション)

リポジトリ直下の `README.md` があれば、v4.7.8 リンクを更新:

```markdown
## Latest Release

**[v4.7.8 — Dwarf Spheroidal Extension](https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.8)** (2026-04-16)

Previous: [v4.7.6](https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.6) (2026-04-15)
```

コミット:
```powershell
git add README.md
git commit -m "Update README with v4.7.8 release link"
git push origin main
```

---

## ステップ 8: WordPress 更新 (並行作業)

GitHub プッシュと並行して:

1. sakaguchi-physics.com (blog_id: 253652152) の該当ページを編集
2. `v478_release/docs/wordpress_v478_main_page.html` の内容を全コピー
3. カスタム HTML ブロックに貼付
4. プレビュー確認 → 公開

---

## ステップ 9: 検証

### GitHub 側
- [ ] https://github.com/sguccibnr32-creator/Public/releases/tag/v4.7.8 が表示される
- [ ] `v478_release/` ディレクトリがリポジトリで閲覧可能
- [ ] README.md がレンダリングされている
- [ ] PDFs がプレビュー可能
- [ ] Release assets に membrane_arxiv_v478.pdf がダウンロード可能

### WordPress 側
- [ ] トップ記事で v4.7.8 更新通知が赤枠で表示
- [ ] 新 Section ⑩ が正しく表示 (表・コードブロック・色枠)
- [ ] 結論が 6 層に更新
- [ ] 確立度整理に A 級新 4 項目が記載

---

## トラブルシューティング

### ファイルサイズが大きすぎる警告

GitHub は単一ファイル 100 MB 超を拒否、50 MB 超で警告。本リリース最大ファイル:
- `section_0_ii_bridge_verification_v1.pdf`: ~920 kB

すべて 1 MB 以下なので問題なし。

### 既存 v4.7.6 ファイルと衝突

`v478_release/` ディレクトリに全て格納しているため、既存ファイルとの衝突は発生しないはず。
`git status` で意図しない変更がある場合は:

```powershell
git diff v478_release/  # 追加分のみ確認
git reset HEAD <意図しない変更のあるファイル>
```

### リモートに v4.7.7 タグが無い

スキップして問題なし。v4.7.7 は内部バージョンで未公開。v4.7.8 は v4.7.6 からの直接ジャンプとして扱う。

### 鍵認証が必要

```powershell
# PAT (Personal Access Token) を使う場合
git remote set-url origin https://<TOKEN>@github.com/sguccibnr32-creator/Public.git

# SSH の場合
git remote set-url origin git@github.com:sguccibnr32-creator/Public.git
```

---

## コマンド一括実行版 (参考)

一気に流したい場合 (PowerShell):

```powershell
cd D:\path\to\Public
git checkout main && git pull origin main
git checkout -b release/v4.7.8

# v478_release/ をコピー (要パス調整)
xcopy /E /I /Y C:\Users\忍\Downloads\v478_release .\v478_release

git add v478_release/
git commit -m "Add v4.7.8 release: Dwarf Spheroidal Extension"
git push -u origin release/v4.7.8

# PR マージは Web UI で実施後、main に戻って:
git checkout main && git pull origin main
git tag -a v4.7.8 -m "v4.7.8: Dwarf Spheroidal Extension"
git push origin v4.7.8

# リリース作成は Web UI (https://github.com/sguccibnr32-creator/Public/releases/new)
```

---

## 連絡

問題があれば以下に記録:
- GitHub Issues: https://github.com/sguccibnr32-creator/Public/issues
- セッション継続ファイル (次回メモ): 作業完了後に新規作成
