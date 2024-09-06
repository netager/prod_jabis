
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
import os
import re
from datetime import datetime

# 처리시간 확인
# ----------
def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

LOG_STATUS = "CONSOLE"
def log_writer(LOG_STATUS, message):
    if LOG_STATUS == "CONSOLE_MONITOR":
        st.write(f"[{get_cur_time()}] {message}")
        print(f"[{get_cur_time()}] {message}")
    elif LOG_STATUS == "CONSOLE":
        print(f"[{get_cur_time()}] {message}")
    elif LOG_STATUS == "MONITOR":
        st.write(f"[{get_cur_time()}] {message}")

log_writer(LOG_STATUS, "(Local_RAG.py) Local_RAG Program Start")

# API KEY 정보로드
load_dotenv()

# 학습 대상 폴더 생성
# -------------------
TRAIN_FILES = "../rag_data/user_train_files"
if not os.path.exists(TRAIN_FILES):
    os.makedirs(TRAIN_FILES)


model_name_path = "../HUGGING_FACE_MODEL/BAAI_bge-m3"
@st.cache_resource()
def embeddings_call():
    return HuggingFaceEmbeddings(
        model_name=model_name_path,
        model_kwargs={"device": "mps"},  # cpu : 'cpu', macOS: 'mps', CUDA: 'cuda'
        # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
        encode_kwargs={
            "normalize_embeddings": True
        },  # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
        # cache_folder='../embedding/model',
    )


def vectorize_file(uploaded_fiels):
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        file_path = f"{TRAIN_FILES}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 단계 1: 문서 로드(Load Documents)
        st.markdown(f'[{get_cur_time()}] {uploaded_file.name}을 읽고 있습니다.')
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        st.markdown(f'[{get_cur_time()}] {uploaded_file.name} 페이지 수: {len(docs)}')

        # 단계 2: 문서 분할(Split Documents)
        st.markdown(f'[{get_cur_time()}] {uploaded_file.name}을 분할하고 있습니다.')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        split_documents = text_splitter.split_documents(docs)
        st.markdown(f'[{get_cur_time()}] {uploaded_file.name} 페이지 수: {len(docs)}, 분할한 문서의 수: {len(split_documents)}')

        ## 일부 필요한 텍스트를 분할된 문서에 추가 및 불필요한 문자 삭제
        final_docs = []

        # 문서만 보여주기 위한 Document 객체 생성 및 저장
        contents = ['보여줘', '알려줘']
        for content in contents:
            title = os.path.splitext(file_path)[0].split('/')[-1]    # 파일명만 추출
            doc = Document(page_content = f"전북은행 {title} {content}")
            doc.metadata['title'] = title        
            doc.metadata['source'] = file_path
            doc.metadata['page'] = 0
            
            final_docs.append(doc)

        doc_cnt = len(final_docs)   # 문서만 보여주기위한 문서 수 

        for doc in split_documents:
            title = os.path.splitext(doc.metadata['source'])[0].split('/')[-1]
            doc.page_content = (
                re.sub(r"(?<!\.)\n", " ", doc.page_content)
                + f"\n\n문서 : 전북은행 {title}"        
            )
            doc.metadata['title'] = title
            final_docs.append(doc)

        st.markdown(f'[{get_cur_time()}] {uploaded_file.name} 분할한 문서의 수: {len(split_documents)}, 보여주기위한 문서수: {doc_cnt}, 처리할 총 문서의 수: {len(final_docs)}')

        # 단계 3: 임베딩(Embedding) 생성
        embeddings_model = embeddings_call()

        # 단계 4: DB 생성(Create DB) 및 저장
        st.markdown(f'[{get_cur_time()}] {uploaded_file.name}을 데이터베이스에 저장하고 있습니다.')
        # 벡터스토어를 생성합니다.
        # vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        # vectorstore = Chroma.from_documents(documents=split_documents, embedding=embeddings, persist_directory="./Chroma_DB/chroma_bank_law_db",
        Chroma.from_documents(documents=split_documents, embedding=embeddings_model,
                                collection_name="bank_law_case",)

        # Chroma.from_documents(documents=split_documents, embedding=embeddings_model,
        #                         persist_directory="../Chroma_DB/chroma_bank_law_db",
        #                         collection_name="bank_law_case",)
        st.markdown(f'---')


# 처리시간 확인
# -------------
def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 헤더 출력
# ---------
head_col1, head_col2 = st.columns([0.9, 0.1], gap="large", vertical_alignment="center")
head_col1.subheader(
    ":blue[Jabis 학습하기 📚]",
    divider="rainbow",
    anchor=None,
    help=None,
)
head_col2.page_link("jabis.py", label="Home", icon="🏠")


# 사이드바 생성
with st.sidebar:
    # 파일 업로드
    uploaded_files = st.file_uploader("학습 파일을 선택하세요", type=["pdf"], accept_multiple_files=True)

# 파일이 업로드 되었을 때
# -----------------------
if uploaded_files:
    st.markdown(f'[{get_cur_time()}] Jabis 학습을 진행합니다.')
    st.markdown(f'---')
    log_writer(LOG_STATUS, "(Local_RAG.py) vectorize_file Start")
    vectorize_file(uploaded_files)

    log_writer(LOG_STATUS, "(Local_RAG.py) 파일 학습 종료")
    st.markdown(f'[{get_cur_time()}] Jabis 학습을 정상 종료합니다. 감사합니다.')