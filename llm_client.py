"""
번역/요약 검증봇 - LangGraph StateGraph
(5-node: intent -> generate -> verify -> (revise <-> verify 사이클) -> finalize)

## 이 프로젝트의 핵심 학습 목표
- LangGraph의 사이클(cycle): 조건에 따라 그래프가 "이전 노드로 되돌아가는" 구조
- 무한 루프를 막기 위한 재시도 횟수(retry counter) 관리

## 노드 구성 (설계 메모 - 세부 구현은 직접)
1. intent_node   : 사용자가 번역/요약 중 뭘 원하는지, 원문이 뭔지 파악
2. generate_node : 초안(번역문 또는 요약문) 생성
3. verify_node   : 원문과 초안을 비교해서 LLM 스스로 비판 (self-critique)
4. verify_router : 비판 결과에 따라 통과(finalize) / 재시도(revise) 분기
5. revise_node   : 비판 내용을 반영해 재생성 -> 다시 verify_node로 (여기가 '사이클' 지점)
6. finalize_node : 통과했거나 재시도 소진 시 최종 답변 정리
"""
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()


# TODO: State 설계
# 힌트: 레시피봇의 RecipeState(TypedDict, total=False)를 참고해서
# 이번엔 어떤 필드가 필요할지 생각해보기. 예:
#   - mode: "translate" 인지 "summarize" 인지
#   - source_text: 원문
#   - draft: 현재까지 생성된 초안
#   - critique: verify_node가 남긴 비판/피드백
#   - is_valid: 검증 통과 여부 (bool)
#   - retry_count: 지금까지 재시도한 횟수
#   - max_retries: 최대 재시도 허용 횟수
#   - answer: 최종 답변
class TranslateState(TypedDict, total=False):
    pass  # TODO: 위 힌트 참고해서 필드 채우기


# TODO: intent_node에서 LLM 구조화 출력에 쓸 Pydantic 모델
# 레시피봇의 IntentResult처럼, mode: Literal["translate", "summarize"] 같은 필드가 필요할 것
class IntentResult(BaseModel):
    pass  # TODO


llm = init_chat_model('gemini-3.1-flash-lite', model_provider='google_genai')
# TODO: intent_node에서 쓸 구조화 출력 LLM
# structured_llm = llm.with_structured_output(IntentResult)


def intent_node(state: TranslateState) -> TranslateState:
    """
    사용자 입력에서 mode(번역/요약)와 source_text(원문)를 뽑아낸다.
    힌트: 레시피봇 intent_node의 structured_llm.invoke(prompt) 패턴 참고.
    """
    # TODO
    pass


def generate_node(state: TranslateState) -> TranslateState:
    """
    mode에 따라 번역문 또는 요약문 '초안'을 생성한다.
    생각해볼 것: 이 노드는 항상 처음부터 새로 생성만 하는지,
    아니면 재시도 흐름에서도 호출되는지 - revise_node와 역할을 어떻게 나눌지 직접 설계.
    """
    # TODO
    pass


def verify_node(state: TranslateState) -> TranslateState:
    """
    source_text와 draft를 LLM에게 같이 주고, 스스로 비판(self-critique)하게 한다.
    - 어떤 기준으로 비판할지 프롬프트 설계가 이 프로젝트의 핵심
      (예: 의미가 왜곡되지 않았는지, 중요한 정보가 누락되지 않았는지 등)
    - 결과를 is_valid(bool) + critique(str) 형태로 구조화해서 state에 반영
    """
    # TODO
    pass


def verify_router(state: TranslateState) -> str:
    """
    verify_node 결과에 따라 다음 노드를 결정하는 라우터.
    - is_valid가 True면 -> finalize
    - is_valid가 False인데 retry_count < max_retries면 -> revise
    - retry_count가 소진됐으면 -> finalize (실패/한계 처리)
    힌트: 레시피봇의 branch_router(state) -> str 패턴 참고.
    """
    # TODO
    pass


def revise_node(state: TranslateState) -> TranslateState:
    """
    critique를 반영해서 draft를 재생성하고, retry_count를 1 증가시킨다.
    이후 그래프는 다시 verify_node로 돌아간다 (사이클).
    """
    # TODO
    pass


def finalize_node(state: TranslateState) -> TranslateState:
    """
    검증을 통과했거나 재시도를 다 쓴 경우, 최종 answer를 정리한다.
    - 통과한 경우: draft를 그대로 혹은 다듬어서 answer로
    - 재시도 소진 + 여전히 invalid인 경우: 사용자에게 뭐라고 안내할지 고민
      (레시피봇 generate_node의 '결과 없음' 처리 참고)
    """
    # TODO
    pass


def build_graph():
    """
    노드/엣지 구성. 여기서 처음으로 '사이클'을 만들게 된다.
    힌트:
      graph.add_edge('revise', 'verify')  # revise 다음 다시 verify로 -> 이게 사이클
      graph.add_conditional_edges('verify', verify_router, {'finalize': 'finalize', 'revise': 'revise'})
    나머지 흐름(START -> intent -> generate -> verify -> ... -> finalize -> END)은
    레시피봇의 build_graph 구조를 참고해서 직접 짜보기.
    """
    # TODO
    pass


def ask(agent, user_message: str, thread_id: str) -> str:
    """레시피봇의 ask() 함수와 동일한 패턴."""
    # TODO
    pass


if __name__ == "__main__":
    # TODO: 테스트 케이스 작성
    # 예: 번역 요청 / 요약 요청 / 일부러 부실한 초안이 나와서 재시도가 도는 경우 등
    pass
