# Update Log v1.8

This version focuses on improving application stability and user experience. Specifically, it resolves the recurring instability of the vector store and enhances the prompt to increase the reliability of the RAG system.

## Core Enhancements

-   **Added and Stabilized "Force Reload" for Vector Store:**
    -   To address the issue where the vector store would intermittently become corrupted and fail document retrieval, a "🔄 Force Reload Vector Store" button was added to the UI.
    -   The initial implementation using `delete_collection` failed in certain inconsistent states. The logic has been improved to use the more robust internal `client.reset()` method to initialize the database. This allows users to reliably reset the vector store with a single click, without restarting the app or manually deleting folders.

-   **Prompt Hardening (Hallucination Suppression):**
    -   To prevent the chatbot from answering questions based on its own knowledge when the answer is not in the provided document (hallucination), the system prompt has been significantly strengthened.
    -   The prompt now explicitly instructs the model to "STRICTLY base the answer on the provided context" and to "NOT use any of its outside knowledge."
    -   It is now forced to reply with a specific phrase ("제공된 문서의 내용으로는 답변할 수 없습니다.") if the answer cannot be found, improving the reliability of the RAG system.

-   **Debugging Logs Restored:**
    -   Per user request, detailed debugging logs that output the results of each stage of the RAG chain (document retrieval, final answer generation) to the terminal have been re-added to facilitate easier troubleshooting.

---

# Update Log v1.7

## New Features

-   **Context Visualization Feature Added:**
    -   Created a new Gradio application, `rag_gradio_llamaparser_with_context_app.py`, to provide an advanced UI.
    -   This application features a split layout: the chat interface on the left and a dedicated panel on the right to display the full text of the documents referenced by the LLM.
    -   The `ask_llm` function was modified to return both the chat history and the retrieved context, allowing the Gradio UI to dynamically update the context panel.
    -   The `ga_prompt` was corrected to ensure proper handling of context as a list of messages, resolving previous `ValueError` issues.

---

# Update Log v1.6

This version includes significant bug fixes, memory management improvements, and UI enhancements for better user experience and RAG chain robustness.

## Bug Fixes

-   **Resolved `NameError: name 'qa_prompt' is not defined`:** Corrected a typo in the final answer generation prompt, changing `qa_prompt` to `ga_prompt`.
-   **Resolved `ValueError: Prompt must accept context as an input variable`:** Reordered the `MessagesPlaceholder` for `context` in the final answer generation prompt (`ga_prompt`) to ensure it correctly receives a list of document messages.

## Improvements & Refinements

-   **Enhanced Conversational Memory Integration:**
    -   Switched from direct Gradio history parsing to using `langchain.memory.ConversationBufferMemory` for managing chat history. This provides a more robust and standard way to handle conversational context within the RAG chain.
    -   The `ask_llm` function now loads chat history from `memory` and saves the current interaction after generating a response.
-   **Improved Example Questions Layout in Gradio UI:**
    -   Restructured the Gradio interface to display example questions in two distinct columns with clear headings ("문서 내용 확인용" and "엑셀표 로드 확인용").
    -   The example questions are now dynamically linked to the main chat input, allowing users to easily populate the input field by clicking on an example.
-   **Updated PDF Source for RAG Chain:**
    -   Changed the primary PDF document for the RAG chain from `data/gemini-2.5-tech_1-10.pdf` to `data/gemini-2.5-tech_1-2.pdf` to focus on a smaller data range.
    -   Updated the corresponding `PARSED_MD_PATH` to `llamaparse_output_gemini_1_2.md`.

## New Tooling

-   **`chunk_visualizer_for_gemini_pdf.py`:**
    -   A new Gradio application created to visualize text splitting for `data/gemini-2.5-tech_1-2.pdf`.
    -   Includes integrated LlamaParse generation logic: if `llamaparse_output_gemini_1_2.md` does not exist, the script will automatically attempt to parse `data/gemini-2.5-tech_1-2.pdf` using LlamaParse and save the output, making the visualizer self-contained.

---

# Update Log v1.5

This version introduces significant enhancements to the RAG application's document processing and conversational capabilities. A new application, `rag_gradio_llamaparser_app.py`, has been created to demonstrate these features.

## Core Enhancements

-   **LlamaParse Integration for Robust Document Processing:**
    -   Switched from `PyMuPDF` to `LlamaParse` for PDF document parsing. `LlamaParse` provides superior extraction of complex layouts, especially tables, ensuring cleaner and more accurate input for the RAG system.
    -   Implemented a caching mechanism: `LlamaParse` is called only once to convert the PDF into a Markdown file (`llamaparse_output_full.md`). Subsequent runs of the application load from this cached Markdown, significantly reducing API calls and startup time.

-   **Conversational Memory with History-Aware Retriever:**
    -   Integrated a `History-Aware Retriever` to enable the RAG chatbot to remember previous turns in the conversation.
    -   The system now intelligently rewrites follow-up questions based on chat history, ensuring more relevant document retrieval and coherent responses.
    -   The RAG chain has been refactored using LangChain Expression Language (LCEL) for a more modular and robust architecture.

## New Application

-   **`rag_gradio_llamaparser_app.py`:** A new Gradio application demonstrating the LlamaParse integration and conversational memory features. This application serves as the primary example for the enhanced RAG pipeline.

---

# Update Log v1.4

This version focuses on a major UI refactoring for simplification and robustness, completes previously planned code updates, and standardizes core library usage.

## UI Refactoring

- **`gr.ChatInterface` Adoption:** The Gradio UI has been refactored to use the high-level `gr.ChatInterface` component. This significantly simplifies the application code by removing the need for manual `gr.Blocks`, `gr.Textbox`, and `gr.Button` event handling (`.click`, `.submit`, `.then`). The `gr.ChatInterface` now natively manages the chat flow, history, and input submission, leading to more robust and maintainable code.

## Core Logic & Library Updates

- **Refactoring Complete (`.invoke()`):** The pending refactoring from v1.2 has been completed. The retriever is now called using `retriever.invoke(query)` instead of the legacy `get_relevant_documents()` method.
- **Standardized PDF Processing (PyMuPDF):** The PDF processing logic now explicitly and consistently uses the `PyMuPDF` (`fitz`) library with modern methods (`page.get_text("text")`) for reliable text extraction.
- **LLM Update:** The model has been updated to `gpt-4o-mini` for improved performance and cost-effectiveness.

---

# Update Log v1.3

This version focuses on fixing a critical runtime error in the self-healing mechanism and addressing library deprecation warnings.

## Bug Fixes

- **`PermissionError: [WinError 32]` on Self-Healing:** Resolved a file locking issue that occurred when the self-healing mechanism tried to delete the ChromaDB directory. The fix replaces the file system-level deletion (`shutil.rmtree`) with a more robust API call (`vectorstore.delete_collection()`) to properly reset the database collection without causing file access conflicts.

## Improvements & Deprecation Fixes

- **Gradio Deprecation:** Set `type='messages'` in the `gr.Chatbot` component to resolve a `UserWarning` and align with modern Gradio standards.
- **LangChain Chroma Deprecation:** Updated the Chroma import from `langchain_community.vectorstores` to `langchain_chroma` to resolve a `LangChainDeprecationWarning` and use the latest package.

---

# Update Log v1.2

This version includes further bug fixes, robustness improvements, and UI enhancements.

## Bug Fixes

- **`SyntaxError: name 'vectorstore' is used prior to global declaration`:** Resolved by correctly placing the `global` declaration for `vectorstore` and `retriever` at the beginning of the `ask_llm` function, and removing redundant declarations.

## Improvements

- **Vector Store Self-Healing:** Implemented a mechanism to automatically detect and re-populate the ChromaDB vector store if no relevant documents are retrieved despite the store appearing populated. This enhances robustness against inconsistent states.
- **UI Enhancement (Permanent Examples):** Modified the Gradio interface to display example questions permanently below the chat input using `gr.Blocks` and `gr.Examples`, improving user guidance.

---

# Update Log v1.1

This version includes bug fixes and minor improvements.

## Bug Fixes

- **`AttributeError: 'dict' object has no attribute 'page_content''`:** Resolved by ensuring that the documents added to the retriever are proper `langchain.schema.Document` objects, not dictionaries (`dict`).

## Improvements

- **PDF Processing Page Limit:** Explicitly limited PDF processing to pages 1-18 in `rag_gradio_app.py` to align with user requirements.
- **ChromaDB Deprecation Warnings:** Acknowledged and noted the deprecation warnings related to `Chroma` import and class usage in LangChain. (No code change, but noted for future updates).

---

# Update Log v1.0

This document summarizes the features implemented in the RAG chatbot up to version 1.0.

## Core Functionality

- **PDF Text Extraction:** Extracts text content from PDF documents.
- **Page Limit:** Processes content from pages 1 to 18 of the PDF.

## RAG System Enhancements

- **Text Splitting:** Implemented advanced text splitting using `RecursiveCharacterTextSplitter` from LangChain.
- **Embedding Generation:** Uses OpenAI's `text-embedding-3-small` model to generate embeddings for text chunks.
- **Vector Store (ChromaDB):** Stores embeddings and text chunks in a persistent ChromaDB instance (`./chroma_db` directory).
- **Caching:** The vector store is cached. On subsequent runs, if the ChromaDB is already populated, it skips reprocessing the PDF and re-embedding, leading to faster startup times.
- **ParentDocumentRetriever:** Utilizes `ParentDocumentRetriever` for efficient retrieval of relevant parent documents based on child chunk similarity.

## User Interface (Gradio)

- **Interactive Chatbot:** Provides a Gradio web interface for asking questions about the PDF content.
- **Korean Language Support:** LLM prompt instructs the model to answer in Korean. Example questions in the UI are also in Korean.