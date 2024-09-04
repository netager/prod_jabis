import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

ctx = get_script_run_ctx()
session_id = ctx.session_id

st.title('Get Session Ids')

st.write(session_id)