import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from datetime import datetime
from dotenv import load_dotenv
from langchain_teddynote import logging

# .env 환경 변수 로딩
# ----------------
load_dotenv()

# LangSmith를 이용하여 LLM 추적
# -------------------------
logging.langsmith("PROD_LLM", set_enable=True)  # enable

from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
ctx = get_script_run_ctx()
session_id = ctx.session_id

def get_cur_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

model_name_path = '../../HUGGING_FACE_MODEL/BAAI_bge-m3'
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

@st.cache_resource()
def llm_model_call():
    return ChatOllama(model="EEVE-Korean-10.8B:latest") 
    # return ChatOllama(model="Llama-3.1-Korean-8B-Instruct_q8_0")
    # llm = ChatOllama(model="qwen2-7b-instruct-q8:latest", temperature=0)    
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0,)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,)

# 세션 기록을 저장1할 딕셔너리



# store = {}
# 세션 ID를 기반으로 세션 기록을 가져오는 함수
DB_INDEX = "MY_FIRST_DB_INDEX"
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = {}
    # st.session_state["chat_memory"] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # st.write(f"[{get_cur_time()}] 1단계 : 문서 로드")
    # # 단계 1: 문서 로드(Load Documents)
    # loader = PDFPlumberLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
    # docs = loader.load()

    # st.write(f"[{get_cur_time()}] 2단계 : 문서 분할")
    # # 단계 2: 문서 분할(Split Documents)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    # split_documents = text_splitter.split_documents(docs)

    # st.write(f"[{get_cur_time()}] 3단계 : 임베딩 로딩")
    # # 단계 3: 임베딩(Embedding) 생성
    # embeddings = embeddings_call()

    # st.write(f"[{get_cur_time()}] 4단계 : 벡터스토어 생성")
    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.

    # db = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    # db.save_local(DB_INDEX)

embeddings = embeddings_call()
vectorstore = FAISS.load_local(DB_INDEX, embeddings, allow_dangerous_deserialization=True)

store = st.session_state["chat_session"]

def get_session_history(session_ids):
    st.write(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = InMemoryChatMessageHistory()
    # else:
        # store[session_ids] = InMemoryChatMessageHistory(messages=messages[:-1])        
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


st.write(f"[{get_cur_time()}] 5단계 : 검색기 생성")
# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

st.write(f"[{get_cur_time()}] 6단계 : 프로프트 생성")
# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Previous Chat History:
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
st.write(f"[{get_cur_time()}] 7단계 : LLM 로드")
llm = llm_model_call()

st.write(f"[{get_cur_time()}] 8단계 : 체인 생성")
# 단계 8: 체인(Chain) 생성
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

st.write(f"[{get_cur_time()}] 단계 : 대화를 기록하는 RAG 체인 생성")
# 대화를 기록하는 RAG 체인 생성
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)

st.title("Chat test")
# 사용자의 입력
# ----------
st.write(f"[{get_cur_time()}] 단계 : 사용자 입력")
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.write(f"[{get_cur_time()}] 단계 : stream 처리")
    response = rag_with_history.stream({"question": user_input},
                                        config={"configurable": {"session_id": session_id}},
                )
    
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

st.write(f"[{get_cur_time()}] 단계 : 종료")