# Google AI Studio API を使ったRAGシステム

このプロジェクトは、Google AI Studio API（Gemini）とLangChainを使用したRAG（Retrieval-Augmented Generation）システムです。

## 機能

- `dataディレクトリ`内のtxtファイルからベクトルデータベースを構築
- Google AI Studio APIを使用した質問応答
- ChromaDBによるベクトル検索
- ソースドキュメントの参照表示

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを作成し、Google AI Studio APIキーを設定してください：

```
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

Google AI Studio APIキーは [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) から取得できます。

### 3. データの準備

`data`ディレクトリを作成し、質問応答の対象となるtxtファイルを配置してください：

```
project/
├── data/
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
├── build_vectordb.py
├── chat.py
├── requirements.txt
└── .env
```

## 使用方法

### 1. ベクトルデータベースの構築

最初に、dataディレクトリ内のtxtファイルからベクトルデータベースを構築します：

```bash
python build_vectordb.py
```

このスクリプトは以下の処理を行います：
- dataディレクトリ内の全てのtxtファイルを読み込み
- ドキュメントをチャンクに分割
- Google AI Studio APIを使用してベクトル化
- ChromaDBにベクトルデータを保存

### 2. チャットの開始

ベクトルデータベースの構築が完了したら、チャットを開始できます：

```bash
python chat.py
```

チャット画面で質問を入力すると、関連するドキュメントを基に回答が生成されます。
終了するには `quit` または `exit` と入力してください。

## ファイル構成

- `build_vectordb.py`: ベクトルデータベース構築用スクリプト
- `chat.py`: チャット機能のメインスクリプト
- `requirements.txt`: 必要なPythonパッケージ
- `.env.example`: 環境変数の設定例
- `README.md`: このファイル

## 注意点

- Google AI Studio APIキーが必要です
- dataディレクトリにtxtファイルが必要です
- 初回実行時は`build_vectordb.py`を先に実行してください
- ベクトルデータベースは`./vectordb`ディレクトリに保存されます

## トラブルシューティング

### よくある問題

1. **APIキーが設定されていない**
   - `.env`ファイルが正しく作成されているか確認
   - Google AI Studio APIキーが有効か確認

2. **txtファイルが見つからない**
   - `data`ディレクトリが存在するか確認
   - txtファイルが正しく配置されているか確認

3. **ベクトルデータベースが見つからない**
   - `build_vectordb.py`を先に実行したか確認
   - `./vectordb`ディレクトリが存在するか確認