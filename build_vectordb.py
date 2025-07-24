import os
import glob
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import hashlib

# 環境変数の読み込み
load_dotenv()

# Google AI Studioの設定
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class VectorDBBuilder:
    def __init__(self, db_path: str = "./chroma_db"):
        """
        ベクトルDBの初期化
        
        Args:
            db_path: ChromaDBの保存パス
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # コレクションの作成または取得
        try:
            self.collection = self.client.create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection(name="document_collection")
        
        # 埋め込みモデルの設定
        self.embedding_model = "models/text-embedding-004"
    
    def load_documents(self, data_dir: str = "./data") -> List[Dict[str, str]]:
        """
        データディレクトリから全てのtxtファイルを読み込む
        
        Args:
            data_dir: データディレクトリのパス
        
        Returns:
            ドキュメントのリスト
        """
        documents = []
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # ドキュメントをチャンクに分割
                chunks = self.split_text(content, chunk_size=500, overlap=50)
                
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()
                    documents.append({
                        "id": doc_id,
                        "content": chunk,
                        "source": file_path,
                        "chunk_index": i
                    })
                    
                print(f"Loaded: {file_path} ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        テキストをチャンクに分割
        
        Args:
            text: 分割するテキスト
            chunk_size: チャンクのサイズ
            overlap: チャンク間のオーバーラップ
        
        Returns:
            チャンクのリスト
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 文の途中で切れないように調整
            if end < len(text):
                last_period = chunk.rfind('。')
                if last_period != -1 and last_period > chunk_size * 0.7:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        テキストの埋め込みを取得
        
        Args:
            texts: 埋め込みを取得するテキストのリスト
        
        Returns:
            埋め込みベクトルのリスト
        """
        embeddings = []
        
        # バッチ処理（Google AI Studioの制限に注意）
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(result['embedding'])
                
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                # エラー時は空の埋め込みを追加
                embeddings.extend([[0.0] * 768 for _ in batch])
        
        return embeddings
    
    def build_vector_db(self):
        """
        ベクトルDBを構築
        """
        print("Loading documents...")
        documents = self.load_documents()
        
        if not documents:
            print("No documents found in data directory.")
            return
        
        print(f"Processing {len(documents)} document chunks...")
        
        # ドキュメントの内容を抽出
        texts = [doc["content"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [{"source": doc["source"], "chunk_index": doc["chunk_index"]} for doc in documents]
        
        # 埋め込みを取得
        print("Getting embeddings...")
        embeddings = self.get_embeddings(texts)
        
        # ChromaDBに追加
        print("Adding to vector database...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"Successfully added {len(documents)} chunks to the vector database.")
        print(f"Database saved at: {self.db_path}")

def main():
    """
    メイン関数
    """
    # データディレクトリの確認
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("Created data directory. Please add .txt files to ./data directory.")
        return
    
    # ベクトルDBの構築
    builder = VectorDBBuilder()
    builder.build_vector_db()

if __name__ == "__main__":
    main()