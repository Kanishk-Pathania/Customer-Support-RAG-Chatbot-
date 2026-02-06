import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain v1.0+ Imports
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Switched to Inference API for "Easily Downloaded" / No-local-compute setup
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq

load_dotenv()


class PolicyRAGSystem:
    def __init__(self, file_path):
        self.file_path = file_path
        # Using Groq (Cloud-based LLM) for speed [cite: 19]
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

        # Engineering Choice: Use Endpoint API instead of local model download
        # This keeps the environment lightweight for the intern assignment
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )
        self.vectorstore = None

    def ingest_data(self):
        """1. Data Preparation"""
        loader = TextLoader(self.file_path)
        documents = loader.load()

        # Engineering Choice: Recursive Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "]
        )
        splits = text_splitter.split_documents(documents)

        # Vector Storage
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        # Return the retriever (Requirement: Semantic Retrieval)
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def get_improved_prompt(self):
        """3. Prompt Engineering (Iteration 2) [cite: 32, 39]"""
        # Strictly grounded prompt to avoid hallucinations [cite: 13, 34]
        system_prompt = (
            "You are a strict Company Policy Assistant. Use ONLY the provided context to answer. "
            "If the answer is not in the context, state: 'Information not found in current policies.' [cite: 35] "
            "Structure your answer with clear headings or bullet points. [cite: 36]\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def build_chain(self):
        # Call ingest_data and get the retriever
        retriever = self.ingest_data()

        context_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user question as a standalone question based on history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, context_q_prompt)
        qa_chain = create_stuff_documents_chain(self.llm, self.get_improved_prompt())

        return create_retrieval_chain(history_aware_retriever, qa_chain)


# 4. Evaluation Module [cite: 41, 46]
if __name__ == "__main__":
    policy_file = "data/company_policies.txt"

    if Path(policy_file).exists():
        rag_app = PolicyRAGSystem(policy_file)
        chain = rag_app.build_chain()

        # Evaluation Set: Answerable, Partially, and Unanswerable [cite: 42, 43, 44, 45]
        eval_questions = [
            ("What is the refund window?", "✅ (Answerable)"),
            ("How much is the late fee for cancellations?", "✅ (Answerable)"),
            ("What is the company's annual revenue?", "❌ (Unanswerable)"),
            ("Do you ship to Mars?", "❌ (Unanswerable/Edge Case)")
        ]

        print("\n--- RAG SYSTEM EVALUATION ---\n")
        for query, category in eval_questions:
            print(f"Testing Category: {category}")
            print(f"Q: {query}")
            result = chain.invoke({"input": query, "chat_history": []})
            print(f"A: {result['answer']}\n")
    else:
        print(f"Error: {policy_file} not found. Please create the file first.")