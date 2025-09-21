# ================================================================
# PDF Conversational RAG + CRAG(Conditional RAG) 통합 버전 (Gradio UI)
# - PDF → LlamaParse(md) → Chroma(Parent/Child) → History-Aware Retrieve
# - CRAG: grade_documents → (generate | transform_query → web_search → generate)
# - 문서 외 지식 금지, 없으면 한국어로 "제공된 문서..." 출력
# - 웹검색: Tavily(선택, 미설정 시 우회)
# ================================================================

import os
import json
import traceback
import gradio as gr
from dotenv import load_dotenv

# Python typing
from typing import Iterable, Optional, Tuple, List
from typing_extensions import TypedDict

# PDF Parser
from llama_parse import LlamaParse

# LangChain Core / OpenAI / Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain Core (Prompts, Messages, Output parsing)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.stores import BaseStore  # ✅ BaseStore는 여기로 이동됨
from langchain_core.messages import BaseMessage

# LangGraph
from langgraph.graph import END, START, StateGraph

# Pydantic (v2)
from pydantic import BaseModel, Field

# (Optional) 웹 검색 툴
from langchain_tavily import TavilySearch


# --- Add: Persistent JSON-backed DocStore for parents ---

# --------------------------
# 유틸: Gradio용 히스토리 변환
# --------------------------
def to_lc_messages(history: List[dict]) -> List:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs


# --------------------------
# 전역 Debug Log 저장소
# --------------------------
debug_logs = []

def log_debug(msg: str):
    debug_logs.append(msg)
    print(msg)  # 콘솔에도 그대로 찍어줌

class JSONDocStore(BaseStore[str, Document]):
    """
    간단한 파일 기반 영구 DocStore.
    - key -> ./parent_store/{key}.json 에 Document 저장
    - ParentDocumentRetriever 가 요구하는 mset/mget/mdelete/yield_keys 구현
    """
    def __init__(self, root_dir: str = "./parent_store"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root_dir, f"{key}.json")

    def mset(self, key_value_pairs: Iterable[Tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    f,
                    ensure_ascii=False,
                )

    def mget(self, keys: Iterable[str]) -> List[Optional[Document]]:
        results: List[Optional[Document]] = []
        for key in keys:
            p = self._path(key)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(
                    Document(
                        page_content=data.get("page_content", ""),
                        metadata=data.get("metadata", {}),
                    )
                )
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
            if not fname.endswith(".json"):
                continue
            key = fname[:-5]  # strip .json
            if prefix is None or key.startswith(prefix):
                yield key


# --------------------------
# 환경변수 로드 & 설정
# --------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")  # 없으면 웹검색 보강은 건너뜀

# --------------------------
# 경로 및 전역 설정
# --------------------------
PDF_PATH = "data/gemini-2.5-tech_1-10.pdf"
PARSED_MD_PATH = "loaddata/llamaparse_output_gemini_1-10.md"
CHROMA_DB_DIR = "./chroma_db10"

# --------------------------
# LLM & 임베딩
# --------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --------------------------
# Text Splitters
# --------------------------
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# --------------------------
# Vector Store (Chroma)
# --------------------------
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# --------------------------
# ParentDocumentRetriever
# --------------------------
# 기존: store = InMemoryStore()
store = JSONDocStore("./parent_store")  # 파일 기반 parent 저장

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2},
)


# --------------------------
# 데이터 로딩 & 벡터DB 적재
# --------------------------
def _vs_count_safe() -> int:
    # 내부 속성 의존을 최소화하는 안전한 카운트 함수
    try:
        return vectorstore._collection.count()  # chroma 내부
    except Exception:
        try:
            # 간단히 비슷문서 조회 시도 (비어있으면 예외 or 빈 결과)
            _ = vectorstore.similarity_search("dummy", k=1)
            # Chroma는 비어있어도 호출이 성공할 수 있으므로 peek 써봄
            return len(vectorstore._collection.peek()["ids"])  # type: ignore
        except Exception:
            return 0

def load_and_populate_vectorstore():
    os.makedirs(os.path.dirname(PARSED_MD_PATH), exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    if _vs_count_safe() > 0:
        print(f"[INFO] Vector store already populated. Count={_vs_count_safe()}")
        return

    # MD 파일 없으면 PDF → LlamaParse → md 저장
    if not os.path.exists(PARSED_MD_PATH):
        print(f"[INFO] '{PARSED_MD_PATH}' not found. Parsing PDF with LlamaParse...")
        if not LLAMA_KEY:
            raise RuntimeError("LLAMA_CLOUD_API_KEY가 없어 PDF 파싱을 수행할 수 없습니다.")
        try:
            parser = LlamaParse(result_type="markdown", api_key=LLAMA_KEY)
            documents = parser.load_data(PDF_PATH)
            md_text = "\n".join([doc.text for doc in documents])
            with open(PARSED_MD_PATH, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"[INFO] Parsed & saved to '{PARSED_MD_PATH}'")
        except Exception as e:
            raise RuntimeError(f"LlamaParse 오류: {e}")

    # md 로드 → Parent retriever에 추가
    print(f"[INFO] Loading markdown from '{PARSED_MD_PATH}'...")
    with open(PARSED_MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 하나의 거대 문서로 추가 → Parent/Child splitter가 내부에서 잘게 쪼갬
    documents = [Document(page_content=text, metadata={"source": PARSED_MD_PATH})]
    retriever.add_documents(documents)
    print(f"[INFO] Vector store populated. Count={_vs_count_safe()}")


# --------------------------
# History-Aware Retriever
# --------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# --------------------------
# 최종 답변(문서 기반만 허용) Chain
# --------------------------
ga_system_prompt = (
    "You are a helpful assistant. Your task is to answer the user's question based on the provided context. "
    "The context may come from PDF documents or from web search results. "
    "If useful information is present in the context (including web search), provide a concise answer. "
    "\n\nContext:\n{context}"
)

ga_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ga_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, ga_prompt)

# --------------------------
# CRAG: 문서 관련성 평가(Structured Output)
# --------------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

grade_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Return 'yes' or 'no'."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_system_prompt),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

# --------------------------
# CRAG: 질문 재작성 (웹검색 친화적)
# --------------------------
rewrite_system = (
    "You are a question re-writer that converts an input question to a better version optimized for web search. "
    "Reason about the underlying semantic intent and produce a clearer query."
)
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system),
        ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved question."),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# --------------------------
# (선택) 웹검색 도구
# --------------------------
web_search_tool: Optional[TavilySearch] = None
if TAVILY_KEY:
    web_search_tool = TavilySearch(k=3)
else:
    print("[WARN] TAVILY_API_KEY 미설정: 웹검색 보강은 생략됩니다.")


# --------------------------
# LangGraph 상태 정의
# --------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    chat_history: List[BaseMessage]  # 항상 HumanMessage/AIMessage만



# --------------------------
# LangGraph 노드 함수
# --------------------------
def node_retrieve(state: GraphState) -> GraphState:
    log_debug("---RETRIEVE---")

    question = state["question"]
    chat_history = state.get("chat_history", [])

    # 원 질문 + 히스토리 출력
    log_debug(f"[DEBUG] Raw Question: {question}")
    if chat_history:
        log_debug(f"[DEBUG] Chat History Count: {len(chat_history)}")
    else:
        log_debug("[DEBUG] No chat history provided.")

    # Child 검색 결과 확인
    child_results = vectorstore.similarity_search(question, k=2)
    log_debug("=== Child 검색 결과 ===")
    for i, d in enumerate(child_results, 1):
        log_debug(f"[Child {i}] Parent ID: {d.metadata.get('doc_id')}")
        log_debug(f"Snippet: {d.page_content[:200]}...\n")

    # Parent 복구 결과 (History-aware retriever 사용)
    docs = history_aware_retriever.invoke(
        {"input": question, "chat_history": chat_history}
    )
    log_debug("=== Parent 복구 결과 ===")
    for i, d in enumerate(docs, 1):
        log_debug(f"[Parent {i}] Source: {d.metadata.get('source', 'N/A')}")
        log_debug(f"Snippet: {d.page_content[:500]}...\n")

    return {
        "documents": docs,
        "question": question,
        "chat_history": chat_history,
        "web_search": "No",
        "generation": "",
    }


def node_grade_documents(state: GraphState) -> GraphState:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs: List[Document] = []
    for d in documents:
        try:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            if score.binary_score.strip().lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        except Exception:
            # 그레이더 실패 시 일단 보수적으로 유지
            filtered_docs.append(d)

    web_search_flag = "Yes" if len(filtered_docs) == 0 else "No"
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search_flag,
        "chat_history": state["chat_history"],
        "generation": state.get("generation", ""),
    }

def node_decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    return "notify_user" if state["web_search"] == "Yes" else "generate"

def node_transform_query(state: GraphState) -> GraphState:
    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {
        "documents": state["documents"],
        "question": better_question,
        "web_search": state["web_search"],
        "chat_history": state["chat_history"],
        "generation": state.get("generation", ""),
    }

def node_web_search(state: GraphState) -> GraphState:
    print("---WEB SEARCH---")
    documents = state["documents"]
    question = state["question"]

    if web_search_tool is None:
        web_results_text = "웹검색 API 키가 설정되지 않아 웹검색을 수행하지 못했습니다."
    else:
        try:
            results = web_search_tool.invoke({"query": question})

            # ✅ 결과가 문자열일 때 대비
            if isinstance(results, str):
                web_results_text = results
            elif isinstance(results, list):
                lines = []
                for r in results:
                    if isinstance(r, dict):  # dict 타입만 처리
                        title = r.get("title", "")
                        url = r.get("url", "")
                        content = r.get("content", "")
                        lines.append(f"[{title}] {url}\n{content}\n")
                    else:
                        lines.append(str(r))  # dict가 아니면 문자열 변환
                web_results_text = "\n---\n".join(lines) if lines else "검색결과가 비어 있습니다."
            else:
                web_results_text = str(results)

        except Exception as e:
            web_results_text = f"웹검색 중 오류: {e}"

    documents = documents + [Document(page_content=web_results_text, metadata={"source": "tavily"})]
    return {
        "documents": documents,
        "question": question,
        "web_search": "No",
        "chat_history": state["chat_history"],
        "generation": state.get("generation", ""),
    }


def node_generate(state: GraphState) -> GraphState:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    # 답변 생성
    answer = question_answer_chain.invoke({
        "input": question,
        "chat_history": chat_history,
        "context": documents
    })

    # 출처 구분
    if any(d.metadata.get("source") == "tavily" for d in documents):
        source_tag = "\n\n[출처: 웹검색 결과]"
    else:
        source_tag = "\n\n[출처: PDF 문서]"

    return {
        "documents": documents,
        "question": question,
        "web_search": "No",
        "chat_history": chat_history,
        "generation": (answer if isinstance(answer, str) else str(answer)) + source_tag,
    }


# --------------------------
# 새 노드: 사용자 알림
# --------------------------
def node_notify_user(state: GraphState) -> GraphState:
    log_debug("---NOTIFY USER---")
    notice = "문서에서 답변을 찾지 못했습니다. 인터넷 검색을 시도합니다."
    chat_history = state.get("chat_history", [])
    chat_history.append(AIMessage(content=notice))  # ✅ dict로만 유지
    return {**state, "chat_history": chat_history}



# --------------------------
# 그래프 구성 & 컴파일
# --------------------------
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("grade_documents", node_grade_documents)
workflow.add_node("generate", node_generate)
workflow.add_node("notify_user", node_notify_user)   # ✅ 추가
workflow.add_node("transform_query", node_transform_query)
workflow.add_node("web_search_node", node_web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 조건 분기 수정
workflow.add_conditional_edges(
    "grade_documents",
    node_decide_to_generate,
    {
        "generate": "generate",            # 문서 관련성 O → 바로 답변
        "notify_user": "notify_user",  # 문서 관련성 X → 먼저 notify_user
    },
)

# 알림 후 웹검색 분기 연결
workflow.add_edge("notify_user", "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()




# --------------------------
# run_crag 수정
# --------------------------
def run_crag(query: str, history: List[dict], show_debug: bool):
    global debug_logs
    debug_logs = []  # 실행할 때마다 초기화

    chat_history_for_chain = to_lc_messages(history or [])
    try:
        final_state = None
        inputs = {"question": query, "chat_history": chat_history_for_chain,
                  "documents": [], "web_search": "No", "generation": ""}
        for step in app.stream(inputs):
            for node_name, node_state in step.items():
                log_debug(f"[TRACE] Node '{node_name}' passed.")
            final_state = node_state

        # 최종 응답
        answer = final_state.get("generation", "제공된 문서의 내용으로는 답변할 수 없습니다.")
        docs: List[Document] = final_state.get("documents", [])
        context_md = "## 참조 문서\n\n"
        if docs:
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "N/A")
                snippet = d.page_content[:500] + ("..." if len(d.page_content) > 500 else "")
                context_md += f"### 문서 {i} (source: {src})\n```\n{snippet}\n```\n\n"
        else:
            context_md += "참조된 문서가 없습니다."

        # 히스토리 추가
        if history is None:
            history = []
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        # 디버그 표시 여부 결정
        debug_output = "### Debug Logs\n```\n" + "\n".join(debug_logs) + "\n```" if show_debug else ""
        return "", history, context_md, debug_output

    except Exception as e:
        err = f"오류 발생: {e}\n{traceback.format_exc()}"
        debug_output = "### 오류\n```\n" + err + "\n```"
        return "", history, "참조된 문서가 없습니다.", debug_output



def force_reload_vectorstore():
    try:
        print("[INFO] Resetting Chroma client...")
        vectorstore._client.reset()  # 전체 컬렉션 초기화
        load_and_populate_vectorstore()
        return "✅ Vector store reloaded successfully!"
    except Exception as e:
        return f"❌ Error during vector store reload: {e}"


# --------------------------
# 초기 적재
# --------------------------
load_and_populate_vectorstore()

# --------------------------
# Gradio UI
# --------------------------
example_questions = [
    "Gemini 2.5 Pro는 Gemini 1.5 Pro와 비교했을 때 어떤 점에서 향상되었나요?",
    "Gemini 2.5 Pro와 Flash는 어떤 종류의 데이터를 처리할 수 있나요?",
    "Gemini 2.5 시리즈의 작은 모델들은 어떤 방식으로 성능을 개선했나요?",
]

with gr.Blocks(theme="soft", title="PDF RAG + CRAG Chatbot") as demo:
    gr.Markdown("# PDF RAG + CRAG Chatbot (LlamaParse / ParentRetriever / History-Aware / Web Search)")
    gr.Markdown("PDF 문서 내용에 대해 질문하세요. 문서에서 못 찾으면 질문 재작성 + (선택)웹검색으로 보강합니다.")

    with gr.Row():
        # ------------------------------
        # 왼쪽: 채팅 영역
        # ------------------------------
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=420, label="Chat", type="messages", value=[])
            msg = gr.Textbox(label="질문을 입력하세요... (Shift+Enter 줄바꿈)")

            gr.Examples(
                examples=example_questions,
                inputs=msg,
                label="예시 질문"
            )

        # ------------------------------
        # 오른쪽: 문서/옵션/디버그 영역
        # ------------------------------
        with gr.Column(scale=2):
            context_display = gr.Markdown(label="LLM 참조 문서 전문/요약")

            with gr.Accordion("⚙️ Advanced Options", open=False):
                show_debug_checkbox = gr.Checkbox(label="Show Debug Logs", value=False)
                debug_panel = gr.Markdown(label="Debug Logs")   # ✅ 디버그 로그 출력 패널
                reload_button = gr.Button("🔄 Force Reload Vector Store")
                reload_status = gr.Markdown()

    # ------------------------------
    # 버튼/이벤트 바인딩
    # ------------------------------
    clear = gr.ClearButton([msg, chatbot, context_display, debug_panel])
    msg.submit(run_crag, [msg, chatbot, show_debug_checkbox],
               [msg, chatbot, context_display, debug_panel])
    reload_button.click(force_reload_vectorstore, outputs=reload_status)

demo.launch()