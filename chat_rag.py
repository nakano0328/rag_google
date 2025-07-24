import os
from typing import List, Dict, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# 環境変数の読み込み
load_dotenv()

# Google AI Studioの設定
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class RAGChat:
    def __init__(self, db_path: str = "./chroma_db"):
        """
        RAGチャットシステムの初期化
        
        Args:
            db_path: ChromaDBのパス
        """
        self.db_path = db_path
        
        # ChromaDBクライアントの初期化
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name="document_collection")
        except:
            raise Exception("Vector database not found. Please run build_vectordb.py first.")
        
        # 埋め込みモデルの設定
        self.embedding_model = "models/text-embedding-004"
        
        # チャットモデルの設定
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # チャット履歴
        self.chat_history = []
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        クエリの埋め込みを取得
        
        Args:
            query: ユーザーのクエリ
        
        Returns:
            埋め込みベクトル
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return [0.0] * 768
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        類似ドキュメントを検索
        
        Args:
            query: 検索クエリ
            n_results: 返す結果の数
        
        Returns:
            類似ドキュメントのリスト
        """
        # クエリの埋め込みを取得
        query_embedding = self.get_query_embedding(query)
        
        # ChromaDBで検索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 結果を整形
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'distance': results['distances'][0][i]
            })
        
        return documents
    
    def generate_response(self, query: str, context_documents: List[Dict]) -> str:
        """
        コンテキストを使用して応答を生成
        
        Args:
            query: ユーザーのクエリ
            context_documents: コンテキストとなるドキュメント
        
        Returns:
            生成された応答
        """
        # コンテキストを作成
        context = "\n\n".join([
            f"[ソース: {doc['source']} - チャンク{doc['chunk_index']}]\n{doc['content']}"
            for doc in context_documents
        ])
        
        # プロンプトを作成
        prompt = f"""以下のコンテキスト情報を使用して、ユーザーの質問に答えてください。
コンテキストに関連する情報がない場合は、その旨を伝えてください。

コンテキスト:
{context}

ユーザーの質問: {query}

回答:"""
        
        try:
            # 応答を生成
            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"エラーが発生しました: {e}"
    
    def chat(self, query: str) -> Tuple[str, List[Dict]]:
        """
        ユーザーの質問に対してRAGを使用して回答
        
        Args:
            query: ユーザーの質問
        
        Returns:
            回答と使用したドキュメントのタプル
        """
        # 類似ドキュメントを検索
        relevant_docs = self.search_similar_documents(query)
        
        # 応答を生成
        response = self.generate_response(query, relevant_docs)
        
        # チャット履歴に追加
        self.chat_history.append({
            "query": query,
            "response": response,
            "sources": relevant_docs
        })
        
        return response, relevant_docs
    
    def interactive_chat(self):
        """
        対話型チャットループ
        """
        print("RAGチャットシステムへようこそ！")
        print("終了するには 'quit' または 'exit' と入力してください。")
        print("-" * 50)
        
        while True:
            # ユーザー入力を取得
            user_input = input("\nあなた: ").strip()
            
            # 終了条件
            if user_input.lower() in ['quit', 'exit', '終了']:
                print("チャットを終了します。")
                break
            
            if not user_input:
                continue
            
            # RAGで回答を生成
            print("\n検索中...")
            response, sources = self.chat(user_input)
            
            # 回答を表示
            print(f"\nアシスタント: {response}")
            
            # ソース情報を表示
            if sources:
                print("\n参考情報:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source['source']} (チャンク{source['chunk_index']}, 距離: {source['distance']:.3f})")

def main():
    """
    メイン関数
    """
    try:
        # RAGチャットシステムを初期化
        rag = RAGChat()
        
        # 対話型チャットを開始
        rag.interactive_chat()
        
    except Exception as e:
        print(f"エラー: {e}")
        print("build_vectordb.pyを実行してベクトルDBを構築してください。")

if __name__ == "__main__":
    main()