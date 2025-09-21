import os
import pathlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 클라이언트를 설정합니다.
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Google GenAI 클라이언트 설정 실패: {e}")
    client = None

def parse_pdf_to_markdown(pdf_path: str, output_dir: str = "loaddata") -> str:
    """
    Gemini 2.5 flash 모델을 사용하여 PDF 파일에서 텍스트와 표를 추출하고,
    그 결과를 마크다운 파일로 저장합니다. (genai.Client() 구문 사용 버전)

    Args:
        pdf_path: 입력 PDF 파일의 경로.
        output_dir: 출력 마크다운 파일을 저장할 디렉토리.

    Returns:
        생성된 마크다운 파일의 경로.
    """
    if not client:
        raise ConnectionError("Google GenAI 클라이언트가 초기화되지 않았습니다. GOOGLE_API_KEY 환경 변수를 확인하세요.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"지정된 PDF 파일을 찾을 수 없습니다: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    pdf_filename = pathlib.Path(pdf_path).stem
    output_filename = f"gemini_parsed_{pdf_filename}.md"
    output_filepath = os.path.join(output_dir, output_filename)

    if os.path.exists(output_filepath):
        print(f"[INFO] 마크다운 파일이 이미 존재하여 파싱을 건너뜁니다: '{output_filepath}'")
        return output_filepath

    file_to_parse = None
    try:
        print(f"[INFO] '{pdf_path}'를 Gemini File API에 업로드 중...")
        file_to_parse = client.files.upload(file=pathlib.Path(pdf_path))

        prompt = """
        I am going to generate a markdown file by extracting the text content from a PDF file. Please proceed in the following order:
        1. The PDF contains various texts, tables, and charts. Extract only the text and tables. Remove headers and footers. Preserve the original language of the document.
        2. Generate a markdown file from the extracted content.
        """

        print(f"[INFO] Gemini 2.5 flash PDF 콘텐츠 생성 중...")
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=[file_to_parse, prompt]
        )

        if not response.candidates:
            feedback = response.prompt_feedback
            raise ValueError(f"응답이 비어있습니다. 안전 설정 때문일 수 있습니다. 피드백: {feedback}")

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"[INFO] 성공적으로 파싱하여 '{output_filepath}'에 저장했습니다.")
        return output_filepath

    except Exception as e:
        raise RuntimeError(f"Gemini 파싱 중 오류 발생: {e}")

    finally:
        if file_to_parse:
            print(f"[INFO] Gemini File API에서 임시 파일 삭제 중...")
            client.files.delete(name=file_to_parse.name)
