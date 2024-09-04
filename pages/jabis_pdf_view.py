import streamlit as st
import base64

# -------------
st.set_page_config(
    page_title="문서 보기",
    # page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)


# st.write(st.query_params)
file_title = st.query_params['title']
file_source = st.query_params['source']
file_page = st.query_params['page']

st.markdown("### 📖  " + file_title)
# st.markdown("### 📖  " + file_source)
# return_jabis = st.button("Return to Jabis", type="primary")    
# if return_jabis:
#     # st.switch_page('pages/test1.py?abc=123')
#     st.switch_page('./pages/jabis_chat.py')


# Opening file from file path
with open(file_source, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

# st.write(base64_pdf)

# Embedding PDF in HTML
pdf_display = F'''<object type="application/pdf" data="data:application/pdf;base64,{base64_pdf}#page={file_page}&navpane=0" width="100%" height="1000">
                <embed src="data:application/pdf;base64,{base64_pdf}#page={file_page}&navpane=0" width="100%" height="1000" type="application/pdf"></embed></object>'''
# pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}#page={file_page}" width="100%" height="1000" type="application/pdf">'
# pdf_display = F'<object src="data:application/pdf;base64,{base64_pdf}#page={file_page}" width="100%" height="1000" type="application/pdf"></object>'
# pdf_display = F'<object src="data:application/pdf;base64,{base64_pdf}#page={file_page}" width="100%" height="1000" type="application/pdf"></object>'
# pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="650" type="application/pdf"></iframe>'
# pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={file_page}" width="100%" height="1000" type="application/pdf"></iframe>'    
# pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'
# pdf_display = F'<iframe src="{file}#toolbar=0 page={page}"></iframe>'
# Displaying File
# return pdf_display
st.markdown(pdf_display, unsafe_allow_html=True)

# st.write(file_source)
# st.write(file_page)


