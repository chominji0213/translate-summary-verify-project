"""
번역/요약 검증봇 - Streamlit UI
레시피봇 app.py의 session_state 패턴을 그대로 따른다.
"""
import uuid
import streamlit as st
import llm_client

st.set_page_config(page_title="번역/요약 검증봇", page_icon="🔁")
st.title("🔁 번역/요약 검증봇")

# TODO: 레시피봇 app.py의 session_state.agent / thread_id / messages 초기화 패턴
#   if "agent" not in st.session_state: ...
#   if "thread_id" not in st.session_state: ...
#   if "messages" not in st.session_state: ...

# TODO: 사이드바에 "새 대화 시작" 버튼 (thread_id, messages 초기화 + st.rerun())

# TODO: 이전 메시지들 렌더링 (for role, content in st.session_state.messages: ...)

# TODO: st.chat_input으로 원문 입력받기
#   생각해볼 것: 번역/요약 중 선택을 어떻게 받을지
#   - intent_node가 텍스트만 보고 판단하게 할지 (예: "이거 요약해줘: ...")
#   - 아니면 라디오 버튼/셀렉트박스로 먼저 모드를 받고 원문만 입력받게 할지
#   두 방식의 장단점을 비교해보고 선택
