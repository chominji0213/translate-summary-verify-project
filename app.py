import uuid
import streamlit as st
import llm_client

st.set_page_config(page_title="번역/요약 검증봇", page_icon="🔁")
st.title("🔁 번역/요약 검증봇")

if "agent" not in st.session_state:
    st.session_state.agent = llm_client.build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("새 대화 시작"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

user_input = st.chat_input("번역하거나 요약할 문장을 입력하세요 (예: '이 문장을 번역해줘: ...', '5줄로 요약해줘: ...')")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("처리 중..."):
            answer = llm_client.ask(st.session_state.agent, user_input, st.session_state.thread_id)
        st.write(answer)

    st.session_state.messages.append(("assistant", answer))
