import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain import hub
from dotenv import load_dotenv

from langchain_teddynote import logging
from time import time
import base64
from streamlit_pdf_viewer import pdf_viewer


# .env 환경 변수 로딩
# ----------------
# load_dotenv()


# LangSmith를 이용하여 LLM 추적
# -------------------------
# logging.langsmith("PROD_LLM", set_enable=True)  # enable



# -----------------------------------------------------------------------------
# Embedding Model Caching
# -----------------------------------------------------------------------------
# Caching Embedding Model
# -----------------------
# model_name_path = '../embedding_model/KR-SBERT-V40K-klueNLI-augSTS'
model_name_path = '../embedding_model/BAAI_bge-m3'

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

# -----------------------------------------------------------------------------
# 함수 Define
# -----------------------------------------------------------------------------
# pdf 출력을 위한 함수
# ----------------
def displayPDF(file, page):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="650" type="application/pdf"></iframe>'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="100%" height="750" type="application/pdf"></iframe>'    
    # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'
    # pdf_display = F'<iframe src="{file}#toolbar=0 page={page}"></iframe>'
    # Displaying File
    # return pdf_display
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



# 체인 생성 함수
def create_chain(prompt_type, user_input):
    # RAG Prompt
    if prompt_type == "은행업무 질의":
        prompt = load_prompt("./prompts/rag.yaml")

    # 프롬프트(기본모드)
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 친절한 20년차 은행원이면서 IT 전문가인 AI 어시스턴트입니다. 다음의 질문에 답변해 주세요.",
                ),
                ("user", """#Question:\n{question}
                            #Answer:\n """),
            ]
        )

    embeddings = embeddings_call()

    # Chroma db Loading
    chroma_db = Chroma(persist_directory="../Chroma_db/chroma_bank_law_db", 
                       embedding_function=embeddings, 
                       collection_name="bank_law_case")

    # chroma_docs = chroma_db.similarit
    # y_search_with_score(user_input, k=3)
    search_results = chroma_db.similarity_search_with_score(user_input, k=3)

    # Retriever 정의
    retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # LLM 모델 선택
    # -----------
    # llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0,)
    # llm = ChatOllama(model="sh2orc-Llama-3.1-Korean-8B-Instruct_q8_0", temperature=0,)
    # llm = ChatOllama(model="sh2orc-Llama-3.1-Korean-8B-Instruct_q8_1", temperature=0,)
    llm = ChatOllama(model="sh2orc-Llama-3.1-Korean-8B-Instruct_q8_2", temperature=0,)

    # llm = ChatOllama(model="Llama-3-BCCard-Kor-8B-q8-0:latest", temperature=0,)
    # llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8:latest", temperature=0)    
    # llm = ChatOllama(model="qwen2-7b-instruct-q8:latest", temperature=0)    
    # llm = ChatOllama(model="llama3:instruct", temperature=0,)
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0,)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    if prompt_type == "은행업무 질의":
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )
    else:

        chain = (
            prompt
            | llm
            | output_parser
        )
    # chain = prompt | llm | output_parser

    return chain, search_results

# 첫인사 하기
def greeting():    
    import random
    initial_messages = ["안녕하세요. 만나뵙게 되어 반갑습니다. 저는 전북은행에서 근무하는 Chat 상담원 Jabis입니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
                        "안녕하세요. 만나뵙게 되어 반갑습니다. 저는 전북은행에서 근무하는 Chat 상담원 Jabis입니다.\n\n 왼쪽 메뉴의 [활용 업무 선택]을 확인하고 궁금한 사항이 있으면 아래에 입력해 주세요.",
                    ]

    random_number = random.randint(0, len(initial_messages) - 1)
    return initial_messages[random_number]


if 'sbstate' not in st.session_state:
    st.session_state.sbstate = 'collapsed'


with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_prompt = st.selectbox("활용 업무 선택 ", ("은행업무 질의", "일반 질의"), index=0)


# Header 출력
# ----------
head_col1, head_col2 = st.columns([0.9, 0.1], gap="large", vertical_alignment="center")
head_col1.subheader('💬 :blue[JABIS(_Jb Ai Business Information System_)]', divider='rainbow', anchor=None, help=None)
# head_col1.header('💬 :blue[JABIS(_JBB Office ChatBot_)]', divider='rainbow', anchor=None, help=None)
head_col2.page_link("jabis.py", label="Home", icon="🏠")

# 대화내용 저장 공간 정의
# -----------------------
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

    # 첫인사 처리
    #------------
    greeting_message = greeting()
    
    st.chat_message("assistant").markdown(greeting_message)    

if clear_btn:
    st.session_state["messages"] = []


# 세션에 저장된 이전 대화 내용을 화면에 출력
# qa_container = st.container(height=700)    

for chat_message in st.session_state["messages"]:
    st.chat_message(chat_message.role).write(chat_message.content)

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약에 사용자 입력이 들어오면
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)

    # chain, document_source, document_page = create_chain(selected_prompt, user_input)
    chain, search_results = create_chain(selected_prompt, user_input)

    # # 화면에 로깅
    # st.info("파일명: " + document_source + "\n페이지: " + str(document_page))    

    # PDF 출력 
    # if(selected_prompt == "은행업무 질의"):
    #     with column_pdf:
    #         displayPDF(document_source, document_page)
    #         st.info("파일명: " + document_source + "\n페이지: " + str(document_page))    


    # 스트리밍 호출
    # response = chain.stream({"question": user_input})
    response = chain.stream(user_input)

    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력을 한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
            # container.write(ai_answer)

        # st.write(len(search_results))
        # for search_result in search_results:
        #     st.write(search_result.score)
        #     st.write(search_result.metadata['page'])

        if selected_prompt == "은행업무 질의":
            # 향후 문서 및 페이지 중복시 중복 제거
            # ------------------------------------
            # test_list = set([(search_result[0].metadata['source'], search_result[0].metadata['page']) for search_result in search_results])
            # st.write(test_list)

            linked_docs = ""
            for search_result in search_results:
            
                # st.write(search_result[1])
                # st.write(search_result[0].metadata['page']+1)

                # linked_docs += f"[{search_result[0].metadata['source']}](http://localhost:8501/jabis_pdf_view?doc_name='{search_result[0].metadata['source']}'&page={search_result[0].metadata['page']})\n\n"

                # linked_docs += f"[{search_result[0].metadata['source']}](http://localhost:8501/jabis_pdf_view?doc_name=ab c&page=1)\n\n"
                # linked_docs += f"['[Naver]()'](http://localhost:8501/jabis_pdf_view?abc=111&page=10)\n\n"

                from urllib.parse import urlencode
                # base_url = 'http://192.168.50.70:8080/jabis_pdf_view'
                base_url = 'http://localhost:8080/jabis_pdf_view'
                params = {
                'source': search_result[0].metadata['source'],
                # 'title': search_result[0].metadata['title'],
                'title': search_result[0].metadata['source'],
                'page': search_result[0].metadata['page']+1
                }
                url_with_params = base_url + '?' + urlencode(params)


                # st.markdown(
                #     f'''<a href="{app_path}/{page}?abc=123" target="_self">goto page 1</a>''',
                #     unsafe_allow_html=True
                # )


                # linked_docs += f"[{search_result[0].metadata['source']}]({url_with_params}) ~ params['page']\n\n"

                # 중복 제거
                # ---------
                linked_docs += f"👉 [{params['title']}]({url_with_params}) ~ <pages : {params['page']}> <{round(search_result[1],3)}>\n\n"

                # app_path = 'http://localhost:8501'
                # page_file_path = 'pages/page1.py'
                # page = page_file_path.split('/')[1][0:-3]  # get "page1"
                # linked_docs += f'''<a href="{app_path}/page1?abc=123" target="_self">goto page 1</a>'''
                # st.write(search_result[0].metadata['source'])
                # st.write(search_result[0].metadata['page']+1)

                # from urllib.parse import urlencode
                # base_url = 'http://localhost:8501/jabis_pdf_view'
                # params = {
                # 'category': './data/jbb/규정/별지 1 공공 마이데이터 활용 동의서(2024. 2. 1 개정).pdf',
                # 'page': 1
                # }    
                # url_with_params = base_url + '?' + urlencode(params)
                
                # page_link(f'http://localhost:8501/jabis_pdf_view?{}', label='Jabis Chat')
            ai_answer = ai_answer + "\n\n 📖 관련 문서 보기\n\n" + linked_docs

            container.markdown(ai_answer, unsafe_allow_html=True)


    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)


# if st.button("Return to Main"):
#     st.session_state.runpage = Page_Main
#     # st.experimental_rerun()
#     st.rerun()