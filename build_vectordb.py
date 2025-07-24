import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 環境変数の読み込み
load_dotenv()

def load_documents_from_directory(directory_path):
    """指定したディレクトリ内の全てのtxtファイルを読み込む"""
    documents = []
    
    # dataディレクトリ内の全てのtxtファイルを取得
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    
    if not txt_files:
        print(f"警告: {directory_path} ディレクトリにtxtファイルが見つかりません。")
        return documents
    
    print(f"{len(txt_files)}個のtxtファイルが見つかりました。")
    
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            file_documents = loader.load()
            documents.extend(file_documents)
            print(f"読み込み完了: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"エラー: {file_path} の読み込みに失敗しました - {e}")
    
    return documents

def split_documents(documents):
    """ドキュメントをチャンクに分割"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"ドキュメントを{len(splits)}個のチャンクに分割しました。")
    
    return splits

def create_vectorstore(documents):
    """ベクトルストアを作成"""
    # Google AI Studio APIキーの確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # Google Generative AI Embeddingsを初期化
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Chromaベクトルストアを作成
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./vectordb"
    )
    
    # Chroma 0.4.x以降では自動で永続化されるため、persist()は不要
    print("ベクトルストアが正常に作成・保存されました。")
    
    return vectorstore

def main():
    """メイン処理"""
    print("RAGシステム用ベクトルDB構築を開始します...")
    
    # データディレクトリの確認
    data_directory = "./data"
    if not os.path.exists(data_directory):
        print(f"エラー: {data_directory} ディレクトリが存在しません。")
        print("dataディレクトリを作成し、txtファイルを配置してください。")
        return
    
    try:
        # 1. ドキュメントの読み込み
        print("1. ドキュメントを読み込んでいます...")
        documents = load_documents_from_directory(data_directory)
        
        if not documents:
            print("読み込むドキュメントがありません。処理を終了します。")
            return
        
        print(f"合計 {len(documents)} 個のドキュメントを読み込みました。")
        
        # 2. ドキュメントの分割
        print("2. ドキュメントを分割しています...")
        splits = split_documents(documents)
        
        # 3. ベクトルストアの作成
        print("3. ベクトルストアを作成しています...")
        vectorstore = create_vectorstore(splits)
        
        print("✅ ベクトルDB構築が完了しました！")
        print("次に chat.py を実行してチャットを開始できます。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()