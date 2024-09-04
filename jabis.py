import streamlit as st
import base64
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="Welcome to Jabis",
    layout="wide",    
    # page_icon="👋",
)

# -----------------------------------------------------------------------------
# Embedding Model Caching
# -----------------------------------------------------------------------------
# Caching Embedding Model
# -----------------------
# model_name_path = '../embedding_model/KR-SBERT-V40K-klueNLI-augSTS'
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

embeddings = embeddings_call()

@st.cache_data
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('./images/jbb_ci.png')
set_background('./images/jbb-building2.webp')
# set_png_as_page_bg('./images/jbb_ci.png')
# set_png_as_page_bg('./images/jbb-building2.webp')

# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
# ctx = get_script_run_ctx()
# session_id = ctx.session_id
# st.write(session_id)

col1, col2, col3, col4 = st.columns([0.7, 0.1, 0.1, 0.1], gap= "small", vertical_alignment="center")
# col1, col2, col3, col4 = st.columns([0.7, 0.1, 0.1, 0.1], gap= "small")
# col1.image("./images/jbb-logo.png", use_column_width="auto" )
col1.image("./images/jbb-logo.png")
col2.write('Overview')
col3.write('Team')
col4.write('About')

main_col1, main_col2 = st.columns([0.6, 0.4], gap="large", vertical_alignment="center")
# main_col1, main_col2 = st.columns([0.6, 0.4], gap="large")

# with main_col1:
#     st.image("./images/jbb-building2.webp", use_column_width="always")

with main_col2:
    st.markdown("# Welcome to Jabis!")
    st.markdown(
    """
    ## Get answers. 
    ## Find inspiration.
    ## Be more productive.
   
    Free to use. Easy to try. Just ask and Jabis can help with writing, learning, brainstorming, and more.
    """
    )

    b_col1, b_col2 , b_col3 = st.columns([0.3, 0.4, 0.2], vertical_alignment="center")
    # b_col1, b_col2 , b_col3 = st.columns([0.3, 0.4, 0.2])

    start_now = b_col2.button("Jabis Start Now", type="primary")    
    if start_now:
        st.switch_page('./pages/jabis_chat.py')


# 향후 개발 되었으면 하는 기능
# st.page_link("pages/a.py", query_params={"update_pwd": True})
# st.page_link("pages/a.py", query_params={"update_pwd": True})
