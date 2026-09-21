# 🔁 번역/요약 검증봇 (translate-summary-verify-project)

LangGraph로 만든 5-노드 에이전트. 사용자가 입력한 문장을 번역하거나 요약하고, 결과를 LLM 스스로 검증(self-critique)해서 문제가 있으면 재시도하는 봇입니다.

"회사 개인공부 트랙" 시리즈(2 → 3 → 4 → 5 노드로 난이도를 올려가며 LangGraph 개념을 하나씩 익히는 프로젝트) 중 다섯 번째 프로젝트이며, 이번 프로젝트의 핵심 학습 목표는 **사이클(cycle)** 입니다.

## 핵심 학습 목표

- **LangGraph 사이클**: 조건에 따라 그래프가 이전 노드로 되돌아가는 구조 (`revise → verify` 엣지)
- **무한 루프 방지**: `retry_count` / `max_retries`로 재시도 횟수를 제한
- **Self-Refine / Self-Critique 패턴**: LLM이 자기 결과물을 스스로 비판하고 고치는 구조

## 그래프 구조

```
START → intent → generate → verify ⇄ revise
                              ↓
                          finalize → END
```

| 노드 | 역할 |
|---|---|
| `intent_node` | 사용자 입력에서 mode(번역/요약), source_text(원문), constraint(제약)를 구조화 출력으로 추출 |
| `generate_node` | mode와 constraint에 맞춰 초안(draft) 생성 |
| `verify_node` | 원문과 draft를 비교해 LLM 스스로 비판 (`is_valid`, `critique`) |
| `verify_router` | is_valid=True → finalize / False + 재시도 여유 → revise / 재시도 소진 → finalize |
| `revise_node` | critique를 반영해 draft 재생성, `retry_count` +1 |
| `finalize_node` | 통과 시 draft를 answer로, 재시도 소진 시 '결과없음' |

## 설계 포인트

**IntentResult에 constraint 필드를 별도로 분리한 이유**
처음엔 `source_text` 하나에 "지시 표현은 제외하고 원문만 추출"하도록 시켰더니, "5줄로 요약해줘" 같은 길이 제약이 케이스에 따라 같이 사라지거나 남는 등 결과가 불안정했습니다. 하나의 필드가 "제거"와 "선택적 보존"을 동시에 판단해야 했기 때문으로 보고, `constraint`를 독립된 필드로 분리해 "원문 내용"과 "부가 조건"의 책임을 나눴습니다. 이후 반복 테스트에서 안정적으로 동작함을 확인했습니다.

**모든 노드에서 구조화 출력(`with_structured_output`)을 쓴 이유**
자유 텍스트로 생성하면 "요청하신 내용을 요약해 드립니다" 같은 인사말/메타 발언이 섞여 나왔습니다. Pydantic 모델의 `Field(description=...)`로 출력 형태를 명시하면 이런 군더더기 없이 필요한 값만 받을 수 있었습니다.

**retry_count / max_retries 초기화 위치**
그래프 내부(intent_node 등)가 아니라 `ask()`에서 `agent.invoke()` 호출 시점에 초기값을 주입하도록 설계했습니다. 재시도 관련 값은 한 번의 대화 요청 단위로 초기화되어야 하는 값이라, 그래프 진입점에서 명시적으로 넣어주는 편이 흐름을 추적하기 쉬웠습니다.

## 배운 점 / 알려진 제약사항

**Self-critique는 만능이 아니다.** `verify_node`가 원문과 결과물을 비교하는 대신, LLM 자신의 배경지식으로 원문 자체의 사실관계를 "정정"하려 든 사례가 있었습니다 (예: 원문에 있는 그대로의 고유명사를 자기 지식과 다르다는 이유로 오류로 지적). 프롬프트에 "배경지식 사용 금지, 원문과 결과물의 문자열만 대조" 같은 강한 제약을 추가해 완화했지만, 완전히 제거됐다고 보장할 수는 없습니다. 검증자(verifier) 역할의 LLM도 스스로 판단 오류를 낼 수 있다는 점을 실무에 적용할 때는 감안해야 합니다.

**broad exception이 실제 버그를 숨길 수 있다.** 개발 중 `revise_node`에서 정의되지 않은 변수를 참조하는 `UnboundLocalError`가 `except Exception`에 조용히 삼켜져, 빈 draft를 반환하는 버그로 이어진 적이 있습니다. 예외 처리 범위를 넓게 잡을수록 디버깅이 어려워질 수 있다는 것을 확인했습니다.

**모드 선택은 UI가 아니라 텍스트에서 판단.** Streamlit UI에서 번역/요약을 라디오 버튼 등으로 먼저 받는 대신, `intent_node`가 사용자 문장만 보고 mode를 추론하도록 설계했습니다. 이미 검증된 로직을 재사용할 수 있고 UI가 단순해지지만, 사용자가 의도를 애매하게 표현하면 mode 판단이 틀릴 가능성은 남아 있습니다.

## 실행 방법

```bash
python -m venv venv
venv\Scripts\activate.bat        # Windows cmd
pip install -r requirements.txt

# .env.example을 .env로 복사 후 GOOGLE_API_KEY 채우기

streamlit run app.py
```

## 기술 스택

- LangGraph (`StateGraph`, 조건부 엣지, 사이클)
- LangChain (`init_chat_model`, `with_structured_output`)
- Pydantic (구조화 출력 스키마)
- Google Gemini (`gemini-3.1-flash-lite`)
- Streamlit (UI)
- SQLite (`SqliteSaver` 체크포인터)
