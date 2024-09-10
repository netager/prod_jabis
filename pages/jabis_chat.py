import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import load_prompt

# from langchain.memory import ConversationBufferMemory
from langchain import hub

from dotenv import load_dotenv

from langchain_teddynote import logging
import time
from datetime import datetime
import base64
from urllib.parse import urlencode

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


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

log_writer(LOG_STATUS, "(jabis_chat.py) Jabis Program Start")

# .env 환경 변수 로딩
# ----------------
load_dotenv()

# LangSmith를 이용하여 LLM 추적
# -------------------------
# logging.langsmith("PROD_LLM", set_enable=True)  # enable


# session id 가져오기
# -----------------
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

ctx = get_script_run_ctx()
session_id = ctx.session_id


# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
# ----------------------------------------------
# if "chat_memory" not in st.session_state:
#     st.session_state["chat_memory"] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
# memory = st.session_state["chat_memory"]

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = {}
chat_store = st.session_state["chat_session"]


def get_session_history(session_ids):
    st.write(f"[대화 세션ID]: {session_ids}")
    if session_ids not in chat_store:  # 세션 ID가 store에 없는 경우
        # chat_store[session_ids] = ChatMessageHistory()
        chat_store[session_ids] = InMemoryChatMessageHistory()
    # else:
    #     store[session_ids] = InMemoryChatMessageHistory(messages=messages[:-1])
    return chat_store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# -----------------------------------------------------------------------------
# Embedding Model Caching
# -----------------------------------------------------------------------------
# Caching Embedding Model
# -----------------------
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


@st.cache_resource()
def llm_model_call():
    return ChatOllama(model="Linkbricks-Llama3.1-Korean-8B-Q8_0:latest")
    # return ChatOllama(model="EEVE-Korean-10.8B:latest")
    # return ChatOllama(model="Llama-3.1-Korean-8B-Instruct_q8_0")
    # llm = ChatOllama(model="qwen2-7b-instruct-q8:latest", temperature=0)
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0,)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,)
    

# -----------------------------------------------------------------------------
# Class Define
# -----------------------------------------------------------------------------
class MyConversationChain(Runnable):

    def __init__(self, llm, prompt, memory, input_key="question"):
        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(self.memory.memory_key)
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer


# -----------------------------------------------------------------------------
# 함수 Define
# -----------------------------------------------------------------------------
# pdf 출력을 위한 함수
# ----------------
def displayPDF(file, page):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="650" type="application/pdf"></iframe>'
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="100%" height="750" type="application/pdf"></iframe>'
    # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'
    # pdf_display = F'<iframe src="{file}#toolbar=0 page={page}"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# 세션에 저장된 이전 대화 내용을 출력하는 함수
# ---------------------------------
def print_saved_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 세션에 대화 내용 저장하는데 ChatMessage 형식으로 저장
# -------------------------------------------
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# retriever(k=3) 결과가 여러개 일때 처리 방법
# Documents의 메타는 제외하고 page_content만 적용
# -----------------------------------------
def content_for_documents(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성 함수
# -----------
def create_chain(prompt_type, user_input):
    # 변수 초기화
    search_results = None

    # RAG Prompt
    if prompt_type == "은행업무 질의":
        prompt = load_prompt("./prompts/rag.yaml")
        # prompt = PromptTemplate.from_template(


        # prompt = PromptTemplate.from_template(
        #     """You are an assistant for question-answering tasks. 
        #     Use the following pieces of retrieved context to answer the question. 
        #     If you don't know the answer, just say that you don't know. 

        #     #Question: 
        #     {question} 

        #     #Previous Chat History:
        #     {chat_history}

        #     #Context: 
        #     {context} 

        #     #Answer:"""
        # )

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "A chat between a curious user and artificial intelligence assistant. The assistant gives simple answer to the user's questions."
        #             # "A chat between a curious user and artificial intelligence assistant. The assistant gives helpfule, detailed, and polite answer to the user's questions."
        #             # "당신은 친절한 20년차 은행원이면서 IT 전문가인 JABIS 입니다. 다음의 질문에 답변해 주세요.",
        #         ),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         (
        #             "user",
        #             """Human: <information>{context}</information>\n\n
        #             #Question: {question}\nAssistant: """
        #         ),

        #     ]
        # )

        # Embedding Call
        log_writer(LOG_STATUS, "(jabis_chat.py) embeddings_call() Start")
        embeddings = embeddings_call()

        # Chroma db Loading
        log_writer(LOG_STATUS, "(jabis_chat.py) Chroma() Loading Start")
        chroma_db = Chroma(
            persist_directory="../Chroma_DB/chroma_bank_law_db",
            embedding_function=embeddings,
            collection_name="bank_law_case",
        )


        # 대상 문서를 보여주기 위해 vector store 검색
        log_writer(LOG_STATUS, "(jabis_chat.py) similarity_search_with_score() k=5")
        search_results = chroma_db.similarity_search_with_score(query=user_input, k=5)

        # Retriever 정의
        retriever = chroma_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 1}
        )
        # retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        # retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 1})
        # search_result = retriever.get_relevant_documents(user_input, k=5)
        # search_results = retriever.invoke(user_input)
        # st.markdown(search_results)

    # 프롬프트(기본모드)
    else:
        search_results = ""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are Jabis of Jeonbuk Bank. A chat between a curious user and artificial intelligence assistant.
                     The assistant gives simple answer to the user's questions."""
                    # Answer in Englist and Korean.""",
                    # "A chat between a curious user and artificial intelligence assistant. The assistant gives helpfule, detailed, and polite answer to the user's questions."
                    # "당신은 친절한 20년차 은행원이면서 IT 전문가인 JABIS 입니다. 다음의 질문에 답변해 주세요.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Human: {question}\nAssistant: "),
            ]
        )

    # LLM 모델 선택
    # -----------
    log_writer(LOG_STATUS, "(jabis_chat.py) llm_model_call() Start")
    llm = llm_model_call()

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    log_writer(LOG_STATUS, "(jabis_chat.py) create chain Start")
    if prompt_type == "은행업무 질의":
        # runnable = RunnablePassthrough.assign(
        #     chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
        # )
                # "chat_history": itemgetter("chat_history"),
        chain_with_history = (
            {
                "context":itemgetter("question") | retriever | content_for_documents,
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | output_parser
        )
        
    else:
        # from langchain_core.runnables import RunnableLambda
        # from operator import itemgetter

        # runnable = RunnablePassthrough.assign(
        #     chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
        # )

        chain = (
            # {
            #     "question": itemgetter("question"),
            #     "chat_history": itemgetter("chat_history"),
            # }
            # | prompt
            prompt
            | llm
            | output_parser
        )

        # for Class : TODO ~ You must implement stream method
        # ---------------------------------------------------
        # conversation_chain = MyConversationChain(llm, prompt, memory)

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    return chain_with_history, search_results
    # return conversation_chain, search_results


# 첫인사말 선택하기
# -----------------
def greeting():
    import random

    initial_messages = [
        "안녕하세요. 만나뵙게 되어 반갑습니다. 저는 전북은행에서 근무하는 Chat 상담원 Jabis입니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
        "좋은날 입니다. Jabis 입니다. 만나뵙게 되어 반갑습니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
        "Jabis 입니다. 날씨가 점점 선선해 지고 있어요. 건강 조심하세요.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
        "운동하기 좋은 날을 기대하며 Jabis가 여러분을 응원합니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
        "Jabis 입니다. 건강을 위해 배드민턴을 권장합니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
    ]

    random_number = random.randint(0, len(initial_messages) - 1)
    return initial_messages[random_number]


# 텍스트 문서 stream 하기
# -----------------------
def stream_data(greeting_message):
    for word in greeting_message.split(" "):
        yield word + " "
        time.sleep(0.05)


# 사용자 화면 작성하기
# --------------------
log_writer(LOG_STATUS, "(jabis_chat.py) 사용자 화면 Start")


if "sbstate" not in st.session_state:
    st.session_state.sbstate = "collapsed"


with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_prompt = st.selectbox(
        "활용 업무 선택 ", ("은행업무 질의", "일반 질의"), index=0
    )


# Header 출력
# ----------
head_col1, head_col2 = st.columns([0.9, 0.1], gap="large", vertical_alignment="center")
head_col1.subheader(
    "💬 :blue[JABIS(_Jb Ai Business Information System_)]",
    divider="rainbow",
    anchor=None,
    help=None,
)
# head_col1.header('💬 :blue[JABIS(_JBB Office ChatBot_)]', divider='rainbow', anchor=None, help=None)
head_col2.page_link("jabis.py", label="Home", icon="🏠")


# 대화내용 저장 공간 정의
# -----------------------
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

    # 첫인사 처리
    # ------------
    greeting_message = greeting()
    st.chat_message("assistant").write_stream(stream_data(greeting_message))


# 이전 대화내용 저장 공간 정의
# -----------------------
# if "chat_store" not in st.session_state:
#     st.session_state["chat_store"] = {}
# store = st.session_state["chat_store"]


# 화면에 출력되는 대화 내용 삭제
# -----------------------
if clear_btn:
    st.session_state["messages"] = []


# 이전의 대화내용을 화면에 출력
# -----------------------
for chat_message in st.session_state["messages"]:
    st.chat_message(chat_message.role).write(chat_message.content)


# 사용자의 입력
# ----------
user_input = st.chat_input("궁금한 내용을 물어보세요!")


# 사용자 질문 입력 및 처리
# -------------------
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)

    # chain, document_source, document_page = create_chain(selected_prompt, user_input)
    log_writer(LOG_STATUS, "(jabis_chat.py) create_chain() Start")
    chain, search_results = create_chain(selected_prompt, user_input)

    # 스트리밍 호출
    log_writer(LOG_STATUS, "(jabis_chat.py) chain.stream() Start")
    response = chain.stream(
        {"question": user_input}, config={"configurable": {"session_id": session_id}}
    )
    log_writer(LOG_STATUS, "(jabis_chat.py) chain.stream() End")

    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력을 한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
            # container.write(ai_answer)

        if selected_prompt == "은행업무 질의":
            # 향후 문서 및 페이지 중복시 중복 제거
            # ------------------------------------
            # test_list = set([(search_result[0].metadata['source'], search_result[0].metadata['page']) for search_result in search_results])
            # st.write(test_list)

            linked_docs = ""
            for search_result in search_results:
                if search_result[1] < 0.8:
                    # base_url = 'http://jabis.jbbank.co.kr:8080/jabis_pdf_view'
                    base_url = "http:/localhost:8501/jabis_pdf_view"
                    # base_url = 'http:/jabis.jbbank.co.kr/jabis_pdf_view'
                    # base_url = 'http:/jabis.jbbank.co.kr/jabis_pdf_view'
                    base_url = "jabis_pdf_view"

                    params = {
                        "source": search_result[0].metadata["source"],
                        "title": search_result[0].metadata["title"],
                        # 'title': search_result[0].metadata['source'],
                        "page": search_result[0].metadata["page"] + 1,
                    }
                    url_with_params = base_url + "?" + urlencode(params)

                    linked_docs += f"👉 [{params['title']}]({url_with_params}) [pages]: {params['page']} [{round(search_result[1],3)}]\n\n"

            ai_answer = ai_answer + "\n\n 📖 관련 문서 보기\n\n" + linked_docs
            container.markdown(ai_answer, unsafe_allow_html=True)

    # 화면에 이전 대화 내용을 출력하기 위해 저장
    # --------------------------------
    add_message("user", user_input)
    add_message("assistant", ai_answer)

    # 이전 대화 내용을 모델에 전달하기 위해 저장
    # --------------------------------
    # st.write(ai_answer) # TODO: delete
    # memory.save_context({"human": user_input}, {"assistant": ai_answer})
    # st.write(memory.load_memory_variables({}))
    log_writer(LOG_STATUS, "(jabis_chat.py) Jabis Chat Program End")
