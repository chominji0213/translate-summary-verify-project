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
    answer: str

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
    """
    if state['is_valid']:
        return {'answer': state['draft']}
    else:
        return {'answer': '결과없음'}


def build_graph():
    """
    노드/엣지 구성
    """
    conn = sqlite3.connect('checkpoint.db', check_same_thread=False)
    memory = SqliteSaver(conn)

    graph = StateGraph(TranslateState)

    graph.add_node('intent', intent_node)
    graph.add_node('generate', generate_node)
    graph.add_node('verify', verify_node)
    graph.add_node('revise', revise_node)
    graph.add_node('finalize', finalize_node)

    graph.add_edge(START, 'intent')
    graph.add_edge('intent', 'generate')
    graph.add_edge('generate', 'verify')
    graph.add_conditional_edges('verify', verify_router, {'revise': 'revise', 'finalize': 'finalize'})
    graph.add_edge('revise', 'verify')
    graph.add_edge('finalize', END)

    agent = graph.compile(checkpointer=memory)

    return agent


def ask(agent, user_message: str, thread_id: str) -> str:
    config = {'configurable': {'thread_id': thread_id}}
    result = agent.invoke({'source_text': user_message, 'retry_count': 0, 'max_retries': 3}, config)

    return result['answer']


if __name__ == "__main__":
    agent = build_graph()
    translate = """
            The dominant sequence transduction models are based on complex recurrent or
            convolutional neural networks that include an encoder and a decoder. The best
            performing models also connect the encoder and decoder through an attention
            mechanism. We propose a new simple network architecture, the Transformer,
            based solely on attention mechanisms, dispensing with recurrence and convolutions
            entirely. Experiments on two machine translation tasks show these models to
            be superior in quality while being more parallelizable and requiring significantly
            less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including
            ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
            our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
            training for 3.5 days on eight GPUs, a small fraction of the training costs of the
            best models from the literature. We show that the Transformer generalizes well to
            other tasks by applying it successfully to English constituency parsing both with
            large and limited training data.
    """

    summarize = """
            미국 10년물 국채금리가 3년 만에 장중 연 5%를 넘어섰다. 2023년에는 5%를 찍은 직후 빠르게 하락했지만, 이번에는 상황이 다르다. 
            고유가와 물가 상승 우려에 미국의 재정 부담, 인공지능(AI) 투자를 위한 회사채 발행까지 겹쳤다. 이번에도 5%가 천장일지, 새로운 바닥일지에 시장의 시선이 쏠리고 있다.
            14일(현지시간) 월스트리트저널(WSJ)에 따르면 미 10년물 국채금리는 장중 5.017%까지 치솟았다(채권 가격은 하락). 이후 저가 매수세가 유입되면서 금리는 4.988%로 내려와 장을 마쳤다. 
            스콧 크로너트 씨티 미국 주식전략가는 파이낸셜타임스(FT)에 5%를 “넘어서는 안 될 선(line in the sand)”이라고 진단했다. 
            몰리 브룩스 TD증권 미국 금리전략가는 “10년물 금리 5%는 투자자들에게 중요한 심리적 기준선”이라며 “일부 투자자들이 저가 매수에 나설 지점으로 정해뒀을 수 있다”고 말했다.
            뉴욕 증시도 일제히 하락했다. 이날 다우존스30 산업평균지수는 0.29%, 스탠더드앤드푸어스(S&P)500 지수는 0.48%, 나스닥종합지수는 0.56% 내렸다. 필라델피아반도체지수는 5.86% 급락했다. 
            10년물 금리 5% 돌파가 부담을 더한 가운데, 인공지능(AI) 속도 조절 우려로 마이크론(-5.25%), 엔비디아(-3.36%) 등 반도체주 매도가 하락을 주도했다.
            관건은 ‘금리 5%’가 이번에도 일시적인 고점에 그칠지다. 2023년 10월에도 10년물 금리는 장중 5%를 넘었지만 그날 다시 5% 아래로 내려왔다. 
            이후 고용과 물가가 둔화하고 미 연방준비제도(Fed)가 긴축을 끝내면서 금리도 하락했다. 이번에는 상황이 다르다는 분석이 나온다. 
            브렌트유가 이날 장중 배럴당 109.80달러까지 치솟는 등 길어지는 중동전쟁에 따른 고유가발 물가 불안이 이어지고 있다.
            구조적인 금리 상승 압력도 만만치 않다. 재정 적자에 따른 국채 공급 확대와 AI 투자를 위한 기업의 대규모 회사채 발행까지 겹쳤다. 
            블룸버그에 따르면 미 국채시장 규모는 2007년 약 4조5000억 달러에서 현재 약 32조 달러로 불어났다. 
            정부와 기업이 동시에 자금을 조달하면서 장기채를 보유하는 투자자들이 더 높은 보상을 요구하고 있다는 분석이다. 
            자크 그리피스 크레디트사이츠 거시전략가는 “금리 상승세가 지속할 수 있는 기저 요인이 많다”며 “10년물 금리가 5.5%까지 오를 수 있다”고 전망했다.
            5%대 금리가 고착되면 실물경제와 증시의 부담도 커진다. 10년물 금리는 부동산 모기지, 자동차 대출과 회사채 등 장기 조달금리의 기준이다. 
            WSJ에 따르면 미국 모기지 금리는 최근 다시 7%에 육박했다. 주식 투자자에게는 안전자산인 국채의 상대적 매력이 커진다. 
            그레그 피터스 PGIM크레딧 공동 최고투자책임자(CIO)는 WSJ에 “금리를 낮출 촉매가 무엇일지 계속 자문하지만, 전통적인 경기침체 말고는 찾기 어렵다”며 “금리가 더 오르거나 높은 수준을 유지할 여건이 상당히 갖춰져 있다”고 말했다.
            이제 시장의 시선은 15~16일(현지시간) 열리는 미 연방공개시장위원회(FOMC)로 향한다.
            시카고상품거래소(CME) 페드워치에 따르면 금리 선물시장은 0.25%포인트 인상 가능성을 90% 넘게 반영하고 있다. 
            Fed로선 금리를 올리면 이미 높아진 가계·기업의 차입 부담을 더 키우지만, 동결하면 물가 대응 의지를 의심받아 장기금리가 오히려 더 뛸 위험이 있다. 
            에드 알후사이니 컬럼비아스레드니들 포트폴리오 매니저는 “금리를 올리지 않으면 대혼란(pandemonium)이 벌어질 것”이라며 “인플레이션(물가 상승) 위험이 커지면서 장기 국채금리가 급등할 것”이라고 짚었다.
    """     

    answer1 = ask(agent, f'{summarize}. 이 원본을 5줄로 요약해줘', "test-thread-1")
    rprint(answer1)