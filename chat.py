import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 環境変数の読み込み
load_dotenv()

def load_vectorstore():
    """保存されたベクトルストアを読み込む"""
    # Google AI Studio APIキーの確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # Embeddingsを初期化
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # 保存されたベクトルストアを読み込み
    vectorstore = Chroma(
        persist_directory="./vectordb",
        embedding_function=embeddings
    )
    
    return vectorstore

def create_qa_chain():
    """QAチェーンを作成"""
    # Google AI Studio APIキーの確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # LLMを初期化（Gemini Pro）
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # ベクトルストアを読み込み
    vectorstore = load_vectorstore()
    
    # プロンプトテンプレートを定義
    prompt_template = """以下の情報を基に、質問に対して正確で詳細な回答を提供してください。
情報に基づいて回答できない場合は、「提供された情報では回答できません」と答えてください。

関連情報:
{context}

質問: {question}

回答:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # RetrievalQAチェーンを作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 関連する上位5つのチャンクを取得
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    """メイン処理"""
    print("🤖 RAGチャットシステムを起動しています...")
    
    # ベクトルストアの存在確認
    if not os.path.exists("./vectordb"):
        print("エラー: ベクトルストアが見つかりません。")
        print("まず build_vectordb.py を実行してベクトルストアを作成してください。")
        return
    
    try:
        # QAチェーンを作成
        qa_chain = create_qa_chain()
        print("✅ RAGシステムの準備が完了しました！")
        print("質問を入力してください。終了するには 'quit' または 'exit' と入力してください。\n")
        
        while True:
            # ユーザーの質問を取得
            question = input("🙋 質問: ").strip()
            
            # 終了条件
            if question.lower() in ['quit', 'exit', '終了', 'やめる']:
                print("チャットを終了します。お疲れさまでした！")
                break
            
            # 空の質問をスキップ
            if not question:
                print("質問を入力してください。")
                continue
            
            try:
                # QAチェーンで回答を生成（新しいinvokeメソッドを使用）
                print("🤔 回答を生成中...")
                result = qa_chain.invoke({"query": question})
                
                # 回答を表示
                print(f"\n🤖 回答: {result['result']}")
                
                # ソースドキュメントがある場合は表示
                if result.get('source_documents'):
                    print(f"\n📚 参照元: {len(result['source_documents'])}個のドキュメントチャンク")
                    for i, doc in enumerate(result['source_documents'][:2], 1):  # 上位2つのソースを表示
                        source = doc.metadata.get('source', '不明')
                        filename = os.path.basename(source) if source != '不明' else '不明'
                        print(f"   {i}. {filename}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                print("もう一度お試しください。")
    
    except Exception as e:
        print(f"初期化エラー: {e}")

if __name__ == "__main__":
    main()