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
from rich import print as rprint
import sqlite3

load_dotenv()

class TranslateState(TypedDict, total=False):
    mode: Literal['translate', 'summarize']
    source_text: str
    draft: str
    constraint: str
    is_valid: bool
    critique: str
    retry_count: int
    max_retries: int
    answer = str

class IntentResult(BaseModel):
    mode: Literal['translate', 'summarize'] = Field(
        description="사용자가 원하는 작업이 번역인지 요약인지 판단한다. "
                    "사용자가 명시적으로 언급하지 않으면 기본값으로 translate를 선택한다."
    )
    source_text: str = Field(
        description="사용자 입력에서 '번역해줘', '요약해줘', '~해줘:' 같은 지시 표현은 제외하고, "
                    "실제로 번역하거나 요약할 대상이 되는 원문 텍스트만 추출한다."
    )
    constraint: str = Field(
        description="사용자가 요청한 추가 조건이나 제약사항만 담는다 "
                    "(예: '5줄로', '간단하게', '격식체로' 같은 길이·형식·톤 요청). "
                    "mode를 나타내는 동사('번역해줘', '요약해줘' 등)는 포함하지 않는다. "
                    "특별한 조건이 없으면 빈 문자열로 둔다."
    )

class DraftResult(BaseModel):
    draft: str = Field(
        description="사용자가 요청한 mode(번역 또는 요약)에 따른 결과물만 담는다. "
                     "'요청하신 내용을 요약해 드립니다' 같은 인사말이나 안내 문구, "
                     "부연 설명은 포함하지 않고, 번역문 또는 요약문 본문만 작성한다."
    )

class VerifyResult(BaseModel):
    is_valid: bool = Field(
        description="원문과 draft를 비교했을 때 문제가 없으면 True, "
                    "문제가 있으면(의미 왜곡, 중요 정보 누락, 제약 미준수 등) False"
    )
    critique: str = Field(
        description = "is_valid가 False일 때, 구체적으로 무엇이 문제인지 서술. "
                    "is_valid가 True면 빈 문자열."
    )

llm = init_chat_model('gemini-3.1-flash-lite', model_provider='google_genai')
structed_llm = llm.with_structured_output(IntentResult)

def intent_node(state: TranslateState) -> TranslateState:
    """
    사용자 입력에서 mode(번역/요약)와 source_text(원문)를 뽑아낸다.
    """
    prompt = f"""
        너는 사용자의 문장이 번역인지 요약인지에 대한 의도를 파악해서 mode에 대해 추출하는 어시스턴트야.

        mode는 다음 두가지 중 하나로 판단해
        "translate" : ~번역해줘
        "summarize" : ~요약해줘     

        사용자 입력: "{state['source_text']}"   
    """

    try:
        raw_response = structed_llm.invoke(prompt)
        result = raw_response

    except Exception as e:
        return {"mode": "translate", 'source_text': ""}

    return {"mode" : result.mode, "source_text": result.source_text, "constraint": result.constraint}


def generate_node(state: TranslateState) -> TranslateState:
    """
    mode에 따라 번역문 또는 요약문 초안을 생성한다.
    """    
    prompt = f"""
            사용자 질문: {state['source_text']}

            너는 사용자에게 번역 혹은 요약을 해주는 친절한 어시스턴스야.

            아래는 조건 따라 맞는 답변을 하면돼.
            - 조건: {state['mode']} , 제약: {state['constraint']}
            - mode가 translate라면 번역을 해주고, mode가 summarize라면 요약을 해주면돼. 
            - 제약을 읽고 해당 제약에 꼭 고려해서 답변해줘
            - 자연스러운 어투로 답변해줘.
            
    """
    try:
        structed_llm = llm.with_structured_output(DraftResult)
        raw_response = structed_llm.invoke(prompt)

    except Exception as e:
        return {'draft': ""}

    return {'draft': raw_response.draft}

    
def verify_node(state: TranslateState) -> TranslateState:
    """
    source_text와 draft를 LLM에게 같이 주고, 스스로 비판(self-critique)하게 한다.
    """
    prompt = f"""
        너는 번역/요약 결과를 엄격하게 검증하는 검수자야.

        [검증 기준]
        1. 의미 왜곡: 원문에 없는 내용이 추가되거나, 원문의 의미가 바뀌지 않았는가
        2. 정보 누락: 원문의 중요한 정보(수치, 날짜, 핵심 주장 등)가 빠지지 않았는가
        3. 제약 준수: 사용자가 요청한 제약("{state['constraint']}")이 있다면 그것을 지켰는가
        4. mode 적합성: mode가 "{state['mode']}"인데, 실제로 그 작업(번역/요약)에 맞는 결과물인가

        문제가 하나라도 있으면 is_valid를 False로 하고, critique에 구체적으로 뭐가 문제인지 적어줘.
        문제가 없으면 is_valid를 True로 하고 critique는 빈 문자열로 둬.

        너는 이 세상에 대해 네가 알고 있는 어떤 배경지식도 이 작업에 절대 사용하지 마.
        원문(source_text)에 적힌 내용이 실제 사실인지 아닌지는 네가 판단할 대상이 아니야.
        너의 유일한 임무는 "원문에 적힌 문장"과 "결과물(draft)에 적힌 문장"을 글자 그대로 대조해서,
        원문에 없는 내용이 결과물에 추가됐는지, 원문에 있는 내용이 결과물에서 빠지거나 바뀌었는지만 확인하는 거야.
        예를 들어 원문에 "A"라고 적혀있으면, 그게 실제로 맞든 틀리든 상관없이 결과물도 "A"라고 돼있으면 통과야.

        [원문]
        {state['source_text']}

        [결과물]
        {state['draft']}

    """
    try:
        structed_llm = llm.with_structured_output(VerifyResult)
        result = structed_llm.invoke(prompt)

    except Exception as e:
            return {'is_valid': False, 'critique': ""}

    return {'is_valid': result.is_valid, 'critique': result.critique}


def verify_router(state: TranslateState) -> str:
    """
    verify_node 결과에 따라 다음 노드를 결정하는 라우터.
    """
    if state['is_valid'] :
        return 'finalize'
    else:
        if state['retry_count'] < state['max_retries']:
            return 'revise'
        else:
            return 'finalize'        


def revise_node(state: TranslateState) -> TranslateState:
    """
    critique를 반영해서 draft를 재생성하고, retry_count를 1 증가시킨다. 이후 그래프는 다시 verify_node로 돌아간다
    """
    prompt = f"""
        너는 이전에 작성한 번역/요약 결과물을 검수 피드백에 맞춰 다시 작성하는 어시스턴트야.

        아래는 원문, 이전에 작성했던 결과물, 그리고 그 결과물의 문제점(피드백)이야.
        피드백에서 지적된 문제를 반드시 반영해서, 결과물을 다시 작성해줘.

        - mode: {state['mode']} (translate면 번역, summarize면 요약)
        - 제약: {state['constraint']}

        [원문]
        {state['source_text']}

        [이전 결과물]
        {state['draft']}

        [문제점(피드백)]
        {state['critique']}

        위 피드백에서 지적된 문제만 정확히 고치고, 나머지는 원문에 충실하게 유지해줘.
        인사말이나 부연 설명 없이 결과물 본문만 작성해줘.
    """

    try:
        structed_llm = llm.with_structured_output(DraftResult)
        raw_response = structed_llm.invoke(prompt)
        new_retry_count = state['retry_count'] + 1
    except Exception as e:
        return {'draft': "", 'retry_count': state['retry_count'] + 1}        


    return {'draft': raw_response.draft, 'retry_count': new_retry_count}
    
    
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


    source = """
    이재명 대통령의 국정수행 지지율이 33.8%를 기록했다는 여론조사 결과가 오늘(14일) 나왔습니다.

리얼미터가 에너지경제신문 의뢰로 지난 7일부터 11일까지 전국 18세 이상 유권자 2515명을 대상으로 조사한 결과 이 대통령의 국정수행 긍정 평가는 33.8%로 집계됐습니다.

이는 직전 리얼미터 조사 대비 3.6%포인트 떨어진 수치입니다.

반면 부정 평가는 직전 조사보다 3.8%포인트 오른 63.3%로 나타났습니다.

긍정 평가와 부정 평가 간 격차는 29.5%포인트로, 오차범위 밖이었습니다.
정당 지지도 조사에서는 국민의힘이 더불어민주당을 앞섰습니다.

지난 10일부터 11일까지 전국 18세 이상 유권자 1003명을 대상으로 진행한 정당 지지도 조사에서 민주당은 36.1%, 국민의힘은 42.1%로 집계됐습니다.

민주당은 직전 조사 대비 5.7%포인트 하락했고 국민의힘은 4.5%포인트 상승했습니다.

양당 격차는 6.0%포인트로, 오차범위 안이었습니다.

이밖에 조국혁신당은 4.7%, 개혁신당 1.3%, 진보당 1.3%, 기타 정당 2.3%로 나타났습니다.

지지 정당이 없는 무당층은 12.2%로 집계됐습니다.

리얼미터는 "국민의힘은 정부의 2기 개각 인선 논란 등 국정 현안에 대한 반사이익 속에 20대 청년층과 충청·PK 지역층이 대거 결집하며 지지율 상승을 이끈 것으로 판단된다"고 했습니다.

민주당에 대해선 "여당으로서 부동산 정책 불확실성과 2기 개각 인사 논란 등 국정 현안 부담이 겹치며 대통령 지지율 하락과 연동해 20대 청년층 및 진보층의 큰 폭 지지 이탈로 지지율이 하락한 것으로 보인다"고 해석했습니다.

두 조사는 모두 무선 자동응답 전화조사 방식으로 진행됐습니다.

표본오차는 대통령 국정수행 평가가 95% 신뢰수준에 ±2.0%포인트, 정당 지지도 조사가 95% 신뢰수준에 ±3.1%포인트입니다.

응답률은 대통령 국정수행 평가가 4.5%, 정당 지지도 조사가 3.6%입니다.



"""



    state = {"source_text": f'{source} 이글 요약해줘', "retry_count": 0, "max_retries": 3}
    intent_result = intent_node(state)
    state.update(intent_result)
    rprint("intent 결과:", state)

    generate_result = generate_node(state)
    state.update(generate_result)
    rprint("generate 결과:", generate_result)


    state['draft'] = "국민의힘 지지율이 25.0%로 민주당(42.1%)에 크게 뒤처진 것으로 나타났습니다."


    verify_result = verify_node(state)
    state.update(verify_result)
    rprint("verify 결과:", verify_result)

    # is_valid가 False일 때만 revise_node로 넘어가는 흐름을 재현
    if not state['is_valid']:
        revise_result = revise_node(state)
        state.update(revise_result)
        rprint("revise 결과:", revise_result)
        rprint("revise 후 retry_count:", state['retry_count'])
