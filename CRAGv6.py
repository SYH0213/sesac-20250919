# =================================================================================================
# 파일 설명: CRAGv6.py
#
# 기능:
# 이 스크립트는 PDF 문서 기반의 질의응답 챗봇을 Gradio 웹 UI로 구현한 것입니다.
# LangChain의 Agent 아키텍처를 기반으로, LLM이 스스로 판단하여 '도구(Tool)'를 동적으로 사용하는
# 고급 RAG(Retrieval-Augmented Generation) 기능을 구현합니다.
#
# 주요 아키텍처 및 기능:
# 1.  **에이전트 (Agent) 기반 동적 파이프라인**:
#     - 고정된 순서(LangGraph) 대신, LLM이 사용자의 질문 의도를 분석하여 가장 적합한 도구를 스스로 선택하고 호출합니다.
#     - 이를 통해 "PDF 내용과 최신 뉴스를 비교해줘"와 같은 복합적인 질문에 유연하게 대응할 수 있습니다.
#
# 2.  **도구 (Tools)**:
#     - **PDF 검색**: `history_aware_retriever`를 사용하여 대화의 맥락을 고려해 PDF 문서에서 관련 정보를 검색하고 답변을 생성합니다.
#     - **웹 검색**: `TavilySearch`를 사용하여 PDF에 없는 최신 정보나 일반적인 질문에 대해 웹에서 정보를 찾아 답변합니다.
#
# 3.  **문서 처리 및 RAG**:
#     - `gemini_parser` (또는 LlamaParse)를 사용해 PDF 문서를 Markdown으로 변환합니다.
#     - `ParentDocumentRetriever`와 `ChromaDB`를 사용하여 효율적인 문서 검색 및 컨텍스트 관리를 수행합니다.
#
# 4.  **메모리 관리**:
#     - Stateless 구조를 유지하며, 매 요청마다 UI로부터 전체 대화 기록을 받아 에이전트의 판단에 활용합니다.
#
# 5.  **UI**:
#     - Gradio를 사용하여 사용자가 쉽게 상호작용할 수 있는 채팅 인터페이스를 제공합니다.
# =================================================================================================

import os
import json
import traceback
import gradio as gr
from dotenv import load_dotenv
from gemini_parser import parse_pdf_to_markdown

# Python typing
from typing import Iterable, Optional, Tuple, List, Any
from typing_extensions import TypedDict

# LangChain - Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.stores import BaseStore
from langchain.schema import Document, StrOutputParser

# LangChain - LLMs & Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# LangChain - Document Processing & Retrieval
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain - Agent & Tools
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_tavily import TavilySearch


# --------------------------
# 유틸: Gradio용 히스토리 변환
# --------------------------
def to_lc_messages(history: List[dict]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

def to_gradio_history(messages: List[BaseMessage]) -> List[dict]:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    return history

# --------------------------
# 전역 Debug Log 저장소
# --------------------------
debug_logs = []

def log_debug(msg: str):
    debug_logs.append(msg)
    print(msg)

# --------------------------
# 파일 기반 영구 DocStore
# --------------------------
class JSONDocStore(BaseStore[str, Document]):
    def __init__(self, root_dir: str = "./parent_store"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root_dir, f"{key}.json")

    def mset(self, key_value_pairs: Iterable[Tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump({"page_content": doc.page_content, "metadata": doc.metadata}, f, ensure_ascii=False)

    def mget(self, keys: Iterable[str]) -> List[Optional[Document]]:
        results: List[Optional[Document]] = []
        for key in keys:
            p = self._path(key)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {})))
            else:
                results.append(None)
        return results

    def mdelete(self, keys: Iterable[str]) -> None:
        for key in keys:
            p = self._path(key)
            if os.path.exists(p):
                os.remove(p)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterable[str]:
        for fname in os.listdir(self.root_dir):
            if fname.endswith(".json"):
                key = fname[:-5]
                if prefix is None or key.startswith(prefix):
                    yield key

# --------------------------
# 환경변수 및 설정
# --------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

PDF_NAME = "gemini-2.5-tech_1-3"
PDF_PATH = f"data/{PDF_NAME}.pdf"
PARSED_MD_PATH = f"loaddata/gemini_parsed_{PDF_NAME}.md"
CHROMA_DB_DIR = "./chroma_db3"

# --------------------------
# 모델 및 임베딩
# --------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --------------------------
# 문서 처리 및 벡터 저장소
# --------------------------
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
store = JSONDocStore("./parent_store3")

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2},
)

def _vs_count_safe() -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0

def load_and_populate_vectorstore():
    os.makedirs(os.path.dirname(PARSED_MD_PATH), exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    if _vs_count_safe() > 0:
        print(f"[INFO] Vector store already populated. Count={_vs_count_safe()}")
        return

    try:
        markdown_file_path = parse_pdf_to_markdown(PDF_PATH, output_dir=os.path.dirname(PARSED_MD_PATH))
    except Exception as e:
        raise RuntimeError(f"Gemini Parser를 사용한 PDF 파싱 중 오류가 발생했습니다: {e}")

    print(f"[INFO] Loading markdown from '{markdown_file_path}'...")
    with open(markdown_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents = [Document(page_content=text, metadata={"source": PARSED_MD_PATH})]
    retriever.add_documents(documents)
    print(f"[INFO] Vector store populated. Count={_vs_count_safe()}")

# --------------------------
# 에이전트 도구(Tools) 정의
# --------------------------

# --- Tool 1: PDF 문서 검색 도구 ---
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise and answer in Korean."
    "\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# RAG Chain을 직접 호출하는 래퍼 함수
# 이 함수는 에이전트의 도구가 호출될 때 현재 대화 기록을 올바르게 참조하기 위해 필요합니다.
def run_rag_chain_with_history(input_dict: dict) -> str:
    # run_agent 함수에서 설정된 전역 대화 기록을 사용
    chat_history = to_lc_messages(current_history_for_tool)
    log_debug(f"---PDF 검색 도구 실행--- (입력: {input_dict.get('input')})")
    result = rag_chain.invoke({"input": input_dict.get("input"), "chat_history": chat_history})
    log_debug(f"---PDF 검색 도구 완료--- (답변: {result.get('answer')})")
    return result.get("answer", "PDF에서 답변을 찾지 못했습니다.")

# --- Tool 2: 웹 검색 도구 ---
web_search_tool = TavilySearch(k=3) if TAVILY_KEY else None

# --- 도구 리스트 생성 ---
tools = []
tools.append(Tool(
    name="pdf_search",
    func=lambda q: run_rag_chain_with_history({"input": q}),
    description="PDF 문서의 내용과 관련된 질문에 답변할 때 사용합니다. 사용자의 질문이 PDF 문서에 대한 것일 경우 이 도구를 우선적으로 사용하세요.",
))
if web_search_tool:
    tools.append(Tool(
        name="web_search",
        func=web_search_tool.invoke,
        description="최신 정보나 PDF 문서에 포함되지 않은 일반적인 주제에 대한 질문에 답변할 때 사용합니다.",
    ))
else:
    print("[WARN] TAVILY_API_KEY가 없어 웹 검색 도구를 비활성화합니다.")

# --------------------------
# 에이전트(Agent) 생성
# --------------------------
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You must answer in Korean. "
               "Analyze the user's question and the conversation history, "
               "and decide whether to use a tool or answer directly. "
               "If the question is about the content of the PDF, use the 'pdf_search' tool. "
               "If the question is about a general topic or requires recent information, use the 'web_search' tool. "
               "For simple greetings or conversational remarks, answer directly without using tools."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, agent_prompt)
# verbose=True로 설정하여 에이전트의 생각 과정을 콘솔에 출력
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --------------------------
# Gradio 실행 함수
# --------------------------
# 도구가 현재 대화 기록에 접근할 수 있도록 전역 변수로 선언
current_history_for_tool = []

def run_agent(query: str, history: List[dict], show_debug: bool):
    global debug_logs, current_history_for_tool
    debug_logs = []
    # 에이전트 실행 전에 현재 히스토리를 전역 변수에 설정
    current_history_for_tool = history or []

    # Gradio의 히스토리를 LangChain 메시지 형식으로 변환
    chat_history_for_agent = to_lc_messages(current_history_for_tool)

    try:
        log_debug("---에이전트 실행 시작---")
        # 에이전트 실행 시, 콘솔에 verbose 출력이 나타남
        response = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history_for_agent
        })
        answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")
        log_debug(f"---에이전트 실행 완료--- (최종 답변: {answer})")

        # Gradio UI에 표시할 히스토리 업데이트
        new_history = current_history_for_tool + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]

        # 디버그 정보 (AgentExecutor의 verbose 출력이 콘솔에 찍히므로 여기서는 간단히 표시)
        context_md = "에이전트가 동적으로 컨텍스트를 판단하고 도구를 사용했습니다. 자세한 과정은 실행 콘솔의 `[agent_executor]` 로그를 확인하세요."
        debug_output = "### Debug Logs\n```\n" + "\n".join(debug_logs) + "\n```" if show_debug else ""

        return "", new_history, context_md, debug_output

    except Exception as e:
        err = f"오류 발생: {e}\n{traceback.format_exc()}"
        log_debug(err)
        debug_output = "### 오류\n```\n" + err + "\n```"
        error_history = current_history_for_tool + [{"role": "user", "content": query}, {"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {e}"}]
        return "", error_history, "참조된 문서가 없습니다.", debug_output

def force_reload_vectorstore():
    try:
        print("[INFO] Resetting Chroma client...")
        vectorstore._client.reset()
        load_and_populate_vectorstore()
        return "✅ Vector store reloaded successfully!"
    except Exception as e:
        return f"❌ Error during vector store reload: {e}"

# --------------------------
# 초기화 및 UI 실행
# --------------------------
load_and_populate_vectorstore()

example_questions = [
    "Gemini 2.5 Pro는 Gemini 1.5 Pro와 비교했을 때 어떤 점에서 향상되었나요?",
    "Gemini 2.5 Pro와 Flash는 어떤 종류의 데이터를 처리할 수 있나요?",
    "오늘 대한민국의 수도 날씨는 어때?",
    "안녕? 오늘 기분 어때?",
]

with gr.Blocks(theme="soft", title="Agent-based PDF Chatbot") as demo:
    gr.Markdown("# Agent-based PDF Chatbot (Dynamic Tool Use)")
    gr.Markdown("에이전트가 질문을 분석하여 PDF 검색, 웹 검색, 또는 직접 답변을 동적으로 결정합니다.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450, label="Chat", type="messages", value=[])
            msg = gr.Textbox(label="질문을 입력하세요... (Shift+Enter 줄바꿈)")
            gr.Examples(examples=example_questions, inputs=msg, label="예시 질문")

        with gr.Column(scale=1):
            context_display = gr.Markdown(label="Agent Context / Debug")
            with gr.Accordion("⚙️ Advanced Options", open=False):
                show_debug_checkbox = gr.Checkbox(label="Show Debug Logs in UI", value=False)
                reload_button = gr.Button("🔄 Force Reload Vector Store")
                reload_status = gr.Markdown()

    clear = gr.ClearButton([msg, chatbot, context_display])
    msg.submit(run_agent, [msg, chatbot, show_debug_checkbox], [msg, chatbot, context_display, context_display])
    reload_button.click(force_reload_vectorstore, outputs=reload_status)

demo.launch()
