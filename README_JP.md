# TideAI

**Tide へようこそ！**  
大規模言語モデルを簡単に管理、微調整、デプロイするためのオールインワンソリューションです。

---

## カラフル AI について

**カラフル AI**は、最先端の AI および生成技術ソリューションを専門としています。私たちの使命は、開発者や企業が大規模言語モデルを管理・デプロイするための直感的なツールを提供することです。**TideAI**を通じて、複雑な AI ワークフローを簡素化し、ユーザーに革新と利便性をもたらすことを目指しています。

---

## 特徴 🚀

1. **LLM モデルのダウンロード** 📥  
   プロジェクトに必要な最新かつ強力な言語モデルを簡単にダウンロードできます。

2. **データセット管理** 📂  
   トレーニングや微調整用のデータセットを効率的に整理・管理します。

3. **モデル操作** ⚙️  
   モデルのトレーニング、微調整、評価を簡単に行えます。

4. **チャットボット** 💬  
   モデルのパフォーマンスをテストし、インタラクティブなチャット機能を体験できます。

---

## 必要条件 🛠️

TideAI を始めるには、以下がインストールされている必要があります：

- [Python](https://www.python.org/) (バージョン 3.8 以上) 🐍
- [Node.js](https://nodejs.org/en) (バージョン 16 以上) 🌐

---

## アカウント要件 🔐

- コードのコラボレーションと管理には GitHub アカウントが必要です。

---

## インストールガイド 📖

**TideAI**をセットアップするには、以下の手順に従ってください：

リポジトリをクローン 📂
git clone https://github.com/ColorfulAIWave/TideAI.git cd TideAI

フロントエンドのインストール 🌐
フロントエンドディレクトリに移動します：

cd Frontend/client Node.js の依存関係をインストールします：

npm install

これでフロントエンドの要件がインストールされました！ 🎉

バックエンドのインストール ⚙️
バックエンドディレクトリに戻ります： cd ../../Backend

Python 仮想環境を作成します： python -m venv venv 仮想環境を有効化します：

Mac/Linux の場合：

source venv/bin/activate

Windows の場合： 仮想環境の有効化には Windows のコマンドプロンプトが必要です。 Tide ディレクトリに移動します：

cd PATH_TO_TIDE_FOLDER 仮想環境を有効化します：

venv\Scripts\activate バックエンドの依存関係をインストールします：

pip install -r requirements.txt

システム要件に基づいて PyTorch をローカルにインストールします： PyTorch インストールガイド

（オプション）追加の依存関係をインストールする場合：

pip install python-multipart

アプリケーションの実行 ▶️
4(A) バックエンドサーバーを起動します：

uvicorn main
--reload

4(B) フロントエンドサーバーを起動します：

cd Frontend/client npm install -g serve serve -s build

---

## お問い合わせ 📞

問題や質問、提案がある場合はお気軽にお問い合わせください：

📧 メール: wave@aiglow.ai  
🌐 ウェブサイト: https://colorful-inc.jp/  
🐞 GitHub の問題: [問題を報告する](https://github.com/ColorfulAIWave/TideAI/issues)
