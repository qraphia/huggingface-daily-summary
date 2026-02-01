---
title: Hugging Face Daily Summary
emoji: 🗞️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Hugging Face Daily Summary

Hugging Face Daily Papers を日付指定で取得し、**抽出要点（非生成）**を表示します。  
必要なら **上位N件だけ** EN→JA 翻訳（NLLB）します（初回はモデルDLで遅い）。

## 1) デプロイ（Hugging Face Spaces）
1. Hugging Face で Space を作る（SDK: Gradio）
2. このリポジトリの `app.py` と `requirements.txt` を Space に置く（git push でもUIアップロードでも可）
3. Build が通れば公開完了

Space の直URL（埋め込み向け）は `https://<space-subdomain>.hf.space` です。

## 2) ローカル実行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
