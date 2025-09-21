# =================================================================================================
# 파일 설명: CRAGv4.py
#
# 기능:
# 이 스크립트는 PDF 문서 기반의 질의응답 챗봇을 Gradio 웹 UI로 구현한 것입니다.
# LangChain과 LangGraph를 기반으로 Corrective RAG (CRAG) 및 일반 RAG 파이프라인을 통합하여,
# 사용자의 질문 의도에 따라 동적으로 응답 방식을 변경하는 고급 기능을 포함합니다.
#
# 주요 아키텍처 및 기능:
# 1.  **문서 처리**:
#     - LlamaParse를 사용해 PDF 문서를 Markdown 형식으로 변환하여 텍스트와 테이블 구조를 정확하게 추출합니다.
#     - 처리된 Markdown은 이후 실행 시 재사용을 위해 캐싱됩니다.
#
# 2.  **RAG (Retrieval-Augmented Generation)**:
#     - `ParentDocumentRetriever`: 문서를 부모/자식 청크로 분할하여 검색 정확도와 컨텍스트 유지의 균형을 맞춥니다.
#     - `ChromaDB`: 벡터 저장소로 사용되며, 임베딩된 문서 청크를 영구적으로 저장합니다.
#     - `History-Aware Retriever`: 대화의 맥락을 이해하여 후속 질문(예: "그건 어때?")을 독립적인 질문으로 재구성합니다.
#
# 3.  **CRAG (Corrective RAG) & 라우팅 (LangGraph 기반)**:
#     - **의도 분류 (Intent Classification)**: 사용자의 입력을 '단순 대화'와 '정보성 질문'으로 분류하는 라우팅 노드를 가장 먼저 실행합니다.
#     - **대화형 응답**: '단순 대화'로 분류되면, RAG 파이프라인을 건너뛰고 LLM이 직접 대화형 답변을 생성합니다.
#     - **문서 관련성 평가**: '정보성 질문'의 경우, 검색된 문서가 질문과 관련이 있는지 LLM이 평가합니다.
#     - **웹 검색 보강**: 관련 문서가 없다고 판단되면, 질문을 웹 검색에 더 적합하게 변형한 후 Tavily API를 통해 웹 검색을 수행하고, 그 결과를 바탕으로 답변을 생성합니다.
#
# 4.  **메모리 관리**:
#     - Stateless 구조: 서버는 대화 기록을 저장하지 않으며, 매 요청마다 Gradio UI(브라우저)로부터 전체 대화 기록을 전달받아 사용합니다.
#     - UI 알림 버그 수정: 웹 검색 시 '인터넷 검색을 시도합니다'와 같은 중간 과정의 알림이 UI에 정상적으로 표시되도록 `run_crag` 함수의 히스토리 관리 로직을 수정했습니다.
#
# 5.  **UI**:
#     - Gradio를 사용하여 사용자가 쉽게 상호작용할 수 있는 채팅 인터페이스를 제공합니다.
# =================================================================================================
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
from gemini_parser import parse_pdf_to_markdown
import pathlib
import shutil

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
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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

# --------------------------------------
# 동적 경로 관리 헬퍼
# --------------------------------------
def get_paths_for_pdf(pdf_filename: str):
    """선택된 PDF 파일명에 따라 동적으로 경로들을 생성합니다."""
    if not pdf_filename:
        return None
    
    base_name = pathlib.Path(pdf_filename).stem
    
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    parsed_md_path = f"loaddata/gemini_parsed_{base_name}.md"
    chroma_db_dir = f"./chroma_db/{base_name}"
    parent_store_dir = f"./parent_store/{base_name}"
    
    return {
        "pdf_path": pdf_path,
        "md_path": parsed_md_path,
        "chroma_dir": chroma_db_dir,
        "store_dir": parent_store_dir,
    }

# --------------------------------------
# Retriever 및 Vectorstore 관리
# --------------------------------------
retriever_cache = {}

def get_retriever_for_pdf(pdf_filename: str):
    """
    선택된 PDF에 대한 retriever를 가져오거나 생성합니다.
    - 캐시 확인 -> 없으면 생성 -> 캐시에 저장
    - Vectorstore가 비어있으면 문서를 파싱하고 DB를 채웁니다.
    """
    if not pdf_filename:
        return None, "PDF 파일을 선택해주세요."

    if pdf_filename in retriever_cache:
        log_debug(f"캐시에서 '{pdf_filename}'에 대한 retriever를 로드합니다.")
        return retriever_cache[pdf_filename], f"'{pdf_filename}'에 대한 준비가 완료되었습니다."

    paths = get_paths_for_pdf(pdf_filename)
    if not paths:
        return None, "경로 생성에 실패했습니다."

    try:
        # 1. Vectorstore 및 Docstore 초기화
        vectorstore = Chroma(persist_directory=paths["chroma_dir"], embedding_function=embeddings)
        store = JSONDocStore(paths["store_dir"])

        # 2. Vectorstore가 비어있는지 확인
        if vectorstore._collection.count() == 0:
            log_debug(f"'{paths['chroma_dir']}'가 비어있습니다. 문서 파싱 및 임베딩을 시작합니다.")
            
            # 3. (필요 시) PDF 파싱
            os.makedirs(os.path.dirname(paths["md_path"]), exist_ok=True)
            markdown_file_path = parse_pdf_to_markdown(paths["pdf_path"], output_dir=os.path.dirname(paths["md_path"]))
            
            with open(markdown_file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            documents = [Document(page_content=text, metadata={"source": markdown_file_path})]
            
            # 4. Retriever 생성 및 문서 추가
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 2},
            )
            retriever.add_documents(documents)
            log_debug(f"Vector store가 성공적으로 생성되었습니다. Count: {vectorstore._collection.count()}")

        else:
            log_debug(f"기존 vector store를 로드합니다. Count: {vectorstore._collection.count()}")
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 2},
            )
        
        # 5. 캐시에 저장
        retriever_cache[pdf_filename] = retriever
        return retriever, f"'{pdf_filename}'에 대한 준비가 완료되었습니다."

    except Exception as e:
        error_msg = f"""'{pdf_filename}' 처리 중 오류 발생: {e}
{traceback.format_exc()}"""
        log_debug(error_msg)
        return None, error_msg




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

    # MD 파일 생성 (기존 LlamaParse 대신 gemini_parser 사용)
    try:
        # gemini_parser.py의 함수를 호출합니다.
        # 이 함수는 내부적으로 파일 존재 여부를 확인하고, 없으면 PDF를 파싱하여 md 파일을 생성한 후,
        # 생성된 마크다운 파일의 최종 경로를 반환합니다.
        markdown_file_path = parse_pdf_to_markdown(PDF_PATH, output_dir=os.path.dirname(PARSED_MD_PATH))
    except Exception as e:
        # GOOGLE_API_KEY가 없거나 다른 오류 발생 시
        raise RuntimeError(f"Gemini Parser를 사용한 PDF 파싱 중 오류가 발생했습니다: {e}")

    # md 로드 → Parent retriever에 추가
    print(f"[INFO] Loading markdown from '{markdown_file_path}'...")
    with open(markdown_file_path, "r", encoding="utf-8") as f:
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

# --------------------------
# 최종 답변(문서 기반만 허용) Chain
# --------------------------
ga_system_prompt = (
    "You are a helpful assistant. Your task is to answer the user's question based on the provided context. "
    "The context may come from PDF documents or from web search results. "
    "If useful information is present in the context (including web search), provide a concise answer. "
    "IMPORTANT: You must answer in Korean."
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
    chat_history: List[BaseMessage]
    intent: str  # "conversational" or "question"
    retriever: Optional[ParentDocumentRetriever]


# --------------------------------------
# 새 노드/Chain: 입력 의도 분류
# --------------------------------------
class ClassifyIntent(BaseModel):
    """"conversational" 또는 "question"으로 사용자 입력의 의도를 분류합니다."""
    intent: str = Field(description="사용자 입력의 의도. 'conversational' 또는 'question' 중 하나여야 합니다.")

classify_system_prompt = """You are a router that classifies the user's input intent. Based on the user's latest message and the previous conversation history, determine if the input is a simple conversation/chit-chat or a question that requires information.
- General greetings like "Hello", "Thank you", "Have a nice day" are 'conversational'.
- Responses to previous answers (e.g., "That's interesting", "I see") are also 'conversational'.
- If the input requires finding information from a PDF document or the web, it is a 'question'.
- If in doubt, classify it as 'question'."""

classify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", classify_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
structured_llm_classifier = llm.with_structured_output(ClassifyIntent)
intent_classifier = classify_prompt | structured_llm_classifier


def node_classify_input(state: GraphState) -> GraphState:
    """사용자 입력의 의도를 분류하여 state에 저장"""
    log_debug("---CLASSIFYING INPUT INTENT---")
    intent_result = intent_classifier.invoke({
        "question": state["question"],
        "chat_history": state.get("chat_history", [])
    })
    log_debug(f"  [Intent] Classified as: {intent_result.intent}")
    return {"intent": intent_result.intent}


# --------------------------------------
# 새 노드/Chain: 단순 대화형 답변 생성
# --------------------------------------
conv_gen_system_prompt = """You are a friendly AI assistant. Respond to the user's message in a natural, conversational way. Do not search for information; just generate a simple response that fits the context of the conversation. IMPORTANT: You must answer in Korean."""

conv_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", conv_gen_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
conversational_chain = conv_gen_prompt | llm | StrOutputParser()


def node_generate_conversational_response(state: GraphState) -> GraphState:
    """단순 대화형 답변을 생성"""
    log_debug("---GENERATING CONVERSATIONAL RESPONSE---")
    generation = conversational_chain.invoke({
        "input": state["question"],
        "chat_history": state.get("chat_history", [])
    })
    return {"generation": generation}


# --------------------------
# LangGraph 노드 함수
# --------------------------
def node_retrieve(state: GraphState) -> GraphState:
    """선택된 retriever를 사용하여 문서를 검색합니다."""
    log_debug("---RETRIEVE---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    retriever = state.get("retriever")

    if not retriever:
        raise ValueError("Retriever가 설정되지 않았습니다. 문서를 먼저 선택하고 로드해주세요.")

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Parent 복구 결과 (History-aware retriever 사용)
    docs = history_aware_retriever.invoke(
        {"input": question, "chat_history": chat_history}
    )
    log_debug(f"--- Retrieved {len(docs)} documents ---")

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

# 1. 새 노드 등록
workflow.add_node("classify_input", node_classify_input)
workflow.add_node("generate_conversational_response", node_generate_conversational_response)

# 2. 기존 노드 등록
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("grade_documents", node_grade_documents)
workflow.add_node("generate", node_generate)
workflow.add_node("notify_user", node_notify_user)
workflow.add_node("transform_query", node_transform_query)
workflow.add_node("web_search_node", node_web_search)

# 3. 시작점 변경
workflow.add_edge(START, "classify_input")

# 4. 의도에 따른 조건부 분기
def decide_flow(state: GraphState) -> str:
    log_debug(f"---DECIDING FLOW BASED ON INTENT: {state['intent']}---")
    if state["intent"] == "conversational":
        return "generate_conversational_response"
    else:
        return "retrieve"

workflow.add_conditional_edges(
    "classify_input",
    decide_flow,
    {
        "generate_conversational_response": "generate_conversational_response",
        "retrieve": "retrieve",
    },
)

# 5. 기존 RAG/CRAG 흐름 연결
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    node_decide_to_generate,
    {
        "generate": "generate",
        "notify_user": "notify_user",
    },
)
workflow.add_edge("notify_user", "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")

# 6. 종료점 연결
workflow.add_edge("generate", END)
workflow.add_edge("generate_conversational_response", END)

app = workflow.compile()




# --------------------------
# Gradio UI 및 이벤트 핸들러
# --------------------------
def get_pdf_list():
    """'data' 디렉토리에서 PDF 파일 목록을 가져옵니다."""
    return [f.name for f in pathlib.Path(DATA_DIR).glob("*.pdf")]

def handle_file_upload(file):
    """파일 업로드 시 호출됩니다."""
    if file is None:
        return gr.update(choices=get_pdf_list())
    
    dest_path = pathlib.Path(DATA_DIR) / pathlib.Path(file.name).name
    shutil.copy(file.name, dest_path)
    
    # 캐시에서 해당 파일이 있다면 삭제하여 리로드를 강제
    if dest_path.name in retriever_cache:
        del retriever_cache[dest_path.name]
        
    return gr.update(choices=get_pdf_list(), value=dest_path.name)

def handle_pdf_selection(pdf_filename, progress=gr.Progress()):
    """드롭다운에서 PDF를 선택했을 때 호출됩니다."""
    if not pdf_filename:
        return "분석할 PDF 파일을 선택해주세요.", ""

    progress(0, desc="문서 처리 준비 중...")
    retriever, msg = get_retriever_for_pdf(pdf_filename)
    progress(1, desc=msg)
    
    return msg, pdf_filename

def run_crag(query: str, history: List[dict], selected_pdf: str, show_debug: bool):
    """
    메인 CRAG 실행 함수. 채팅 메시지 제출 시 호출됩니다.
    UI의 모든 입력을 받아 LangGraph를 실행하고 결과를 스트리밍으로 반환합니다.
    """
    global debug_logs
    debug_logs = []

    # --- 입력 유효성 검사 ---
    if not query:
        history.append({"role": "user", "content": ""})
        history.append({"role": "assistant", "content": "질문을 입력해주세요."})
        yield "", history, "질문을 입력해주세요.", ""
        return

    if not selected_pdf:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "먼저 분석할 PDF 문서를 선택해주세요."})
        yield "", history, "PDF 문서를 선택해주세요.", ""
        return

    # --- Retriever 준비 ---
    retriever, msg = get_retriever_for_pdf(selected_pdf)
    if not retriever:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"문서 준비 중 오류가 발생했습니다: {msg}"})
        yield "", history, msg, ""
        return

    # --- LangGraph 실행 ---
    chat_history_for_chain = to_lc_messages(history or [])
    
    try:
        inputs = {
            "question": query, 
            "chat_history": chat_history_for_chain,
            "retriever": retriever
        }
        
        # 사용자 질문을 히스토리에 먼저 추가
        history.append({"role": "user", "content": query})
        
        # 스트리밍 실행 및 답변 생성
        generation = ""
        final_state = {}
        for step in app.stream(inputs):
            node_name = list(step.keys())[0]
            final_state = step[node_name]
            log_debug(f"[TRACE] Node '{node_name}' passed.")
            
            # 웹 검색 시 중간 알림
            if node_name == "grade_documents" and final_state.get("web_search") == "Yes":
                history.append({"role": "assistant", "content": "문서에서 답변을 찾지 못했습니다. 인터넷 검색을 시도합니다."})
                yield "", history, "웹 검색을 시작합니다...", ""

            if "generation" in final_state and final_state["generation"]:
                 generation = final_state["generation"]

        # 최종 답변을 히스토리에 추가/업데이트
        if history[-1]["role"] == "assistant": # 웹 검색 알림이 있었던 경우
            history[-1]["content"] = generation
        else:
            history.append({"role": "assistant", "content": generation})

        # --- 결과 표시 ---
        docs: List[Document] = final_state.get("documents", [])
        context_md = "## 참조 문서\n\n"
        if docs:
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "N/A")
                snippet = d.page_content[:500] + ("..." if len(d.page_content) > 500 else "")
                context_md += f"### 문서 {i} (source: {src})\n```\n{snippet}\n```\n\n"
        else:
            context_md += "참조된 문서가 없습니다."

        debug_output = "### Debug Logs\n```\n" + "\n".join(debug_logs) + "\n```" if show_debug else ""
        
        yield "", history, context_md, debug_output

    except Exception as e:
        err = f"오류 발생: {e}\n{traceback.format_exc()}"
        debug_output = "### 오류\n```\n" + err + "\n```"
        history.append({"role": "assistant", "content": "죄송합니다. 답변 생성 중 오류가 발생했습니다."})
        yield "", history, "오류가 발생했습니다.", debug_output


# --------------------------
# Gradio UI 구성
# --------------------------
example_questions = [
    "Gemini 2.5 Pro는 Gemini 1.5 Pro와 비교했을 때 어떤 점에서 향상되었나요?",
    "Gemini 2.5 Pro와 Flash는 어떤 종류의 데이터를 처리할 수 있나요?",
    "Gemini 2.5 시리즈의 작은 모델들은 어떤 방식으로 성능을 개선했나요?",
]

with gr.Blocks(theme="soft", title="Dynamic PDF RAG + CRAG Chatbot") as demo:
    gr.Markdown("# Dynamic PDF RAG + CRAG Chatbot")
    gr.Markdown("좌측 상단에서 분석할 PDF를 선택하거나 새 파일을 업로드하세요. 문서가 준비되면 질문을 시작할 수 있습니다.")

    # 현재 선택된 PDF 파일명을 저장하기 위한 상태
    selected_pdf_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            # --- 파일 관리 섹션 ---
            with gr.Accordion("1. 문서 선택 및 관리", open=True):
                pdf_selector = gr.Dropdown(
                    label="분석할 PDF 문서 선택",
                    choices=get_pdf_list(),
                    interactive=True
                )
                upload_button = gr.UploadButton("PDF 업로드", file_types=[".pdf"])
                status_display = gr.Markdown("대기 중...")

            # --- 채팅 섹션 ---
            chatbot = gr.Chatbot(height=420, label="Chat", type="messages", value=[])
            msg = gr.Textbox(label="질문을 입력하세요... (Shift+Enter 줄바꿈)")
            
            gr.Examples(
                examples=example_questions,
                inputs=msg,
                label="예시 질문"
            )

        with gr.Column(scale=2):
            context_display = gr.Markdown(label="LLM 참조 문서")
            with gr.Accordion("⚙️ Advanced Options", open=False):
                show_debug_checkbox = gr.Checkbox(label="Show Debug Logs", value=False)
                debug_panel = gr.Markdown(label="Debug Logs")

    # --- 이벤트 핸들러 바인딩 ---
    clear = gr.ClearButton([msg, chatbot, context_display, debug_panel, status_display])

    # 1. 파일 업로드 시: 파일을 서버에 저장하고, 드롭다운 목록을 갱신
    upload_button.upload(handle_file_upload, inputs=[upload_button], outputs=[pdf_selector])

    # 2. 드롭다운에서 PDF 선택 시: 해당 PDF에 대한 retriever를 준비/로드
    pdf_selector.change(
        handle_pdf_selection, 
        inputs=[pdf_selector], 
        outputs=[status_display, selected_pdf_state]
    )

    # 3. 메시지 전송 시: CRAG 파이프라인 실행
    msg.submit(
        run_crag, 
        [msg, chatbot, selected_pdf_state, show_debug_checkbox],
        [msg, chatbot, context_display, debug_panel]
    )

demo.launch()