from dotenv import load_dotenv
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st

# api key값 가져오기 (보안)
load_dotenv()
API_KEY=os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container #streamlit container의미
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token #텍스트를 줄줄이 받아서 추가
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""
content="""
# 프롬프트 엔지니어링

# 필요성

프롬프트 엔지니어링이란 인공지능(AI) 언어 모델에게 특정한 출력을 유도하기 위해 입력 텍스트(프롬프트)를 세심하게 설계하는 과정을 의미합니다. 이는 AI가 다양한 작업을 수행하도록 하기 위해 필수적인 과정으로, 최근 OpenAI의 GPT와 같은 고급 언어 모델의 등장으로 그 중요성이 더욱 부각되고 있습니다. 프롬프트 엔지니어링의 필요성은 다음과 같은 몇 가지 이유에서 기인합니다:

### **1. AI의 성능 최적화**

프롬프트를 효과적으로 설계함으로써 AI 모델의 응답 품질과 정확성을 극대화할 수 있습니다. 적절하고 명확한 지시를 제공함으로써 모델이 요구하는 작업을 정확히 이해하고 수행할 수 있도록 돕습니다.

### **2. 특정 작업 맞춤화**

언어 모델은 다양한 종류의 문제를 해결할 수 있지만, 모든 작업에 대해 동일한 수준의 성능을 보이지는 않습니다. 프롬프트 엔지니어링을 통해 모델에 특정 작업에 대한 충분한 정보와 맥락을 제공함으로써, 특정 작업에 대한 모델의 성능을 맞춤화하고 최적화할 수 있습니다.

### **3. 창의적인 문제 해결**

프롬프트 엔지니어링은 모델이 예상치 못한 방식으로 문제를 해결하도록 유도할 수 있습니다. 다양한 프롬프트 전략을 실험하면서 모델이 제공할 수 있는 창의적이고 혁신적인 해결책을 탐색할 수 있습니다.

### **4. 오해의 최소화**

언어 모델은 때때로 잘못된 정보를 제공하거나 사용자의 의도를 잘못 해석할 수 있습니다. 프롬프트를 신중하게 설계함으로써 이러한 오해의 가능성을 최소화하고, 모델이 더 정확하고 관련성 높은 응답을 제공하도록 할 수 있습니다.

### **5. 안전성 및 윤리적 고려**

AI 모델이 불쾌감을 주거나 윤리적으로 문제가 될 수 있는 내용을 생성하지 않도록 하기 위해서도 프롬프트 엔지니어링이 중요합니다. 모델이 특정한 유형의 내용을 생성하거나 피하는 방식을 조절하기 위해 세심한 프롬프트 설계가 필요합니다.

### **결론**

프롬프트 엔지니어링은 AI 언어 모델의 성능을 최대한 활용하고, 특정 작업에 맞춤화된 응답을 얻기 위한 핵심적인 과정입니다. 이를 통해 모델의 정확성과 창의성을 향상시키고, 오해를 최소화하며, 안전하고 윤리적인 AI 사용을 보장할 수 있습니다.

# 프롬프트 엔지니어링 요령

프롬프트 엔지니어링은 언어 모델, 특히 OpenAI의 GPT와 같은 고급 모델을 사용할 때 중요한 역할을 합니다. 여기 몇 가지 유용한 프롬프트 엔지니어링 요령을 소개합니다:

### **1. 분명한 목표 설정**

- 프롬프트를 작성하기 전에, 모델에게 어떤 종류의 응답을 기대하는지 분명히 정의합니다.
- 구체적이고 명확한 지시를 제공하여 모델이 원하는 출력을 생성할 수 있도록 합니다.

### **2. 예시 사용**

- 모델에게 원하는 응답 형식을 보여주는 예시를 포함시킵니다.
- "다음과 같이 답변해주세요:"와 같은 지시문 뒤에 예시를 추가하여 모델이 응답 형식을 이해하도록 돕습니다.

### **3. 적절한 상세도 제공**

- 너무 모호하거나 너무 구체적인 프롬프트는 원하지 않는 결과를 초래할 수 있습니다.
- 원하는 출력에 필요한 충분한 정보와 맥락을 제공합니다.

### **4. 반복적인 피드백 활용**

- 초기 응답을 바탕으로 프롬프트를 조정하며 원하는 결과를 얻을 때까지 실험합니다.
- 모델의 응답에서 배울 수 있는 점을 찾아 프롬프트를 개선합니다.

### **5. 역할 설정**

- 프롬프트에서 모델의 역할을 설정함으로써 특정한 태도나 전문성 수준을 유도합니다.
- 예를 들어, "당신은 전문적인 과학자입니다:"와 같이 시작하는 프롬프트는 과학적인 답변을 유도할 수 있습니다.

### **6. 명확한 종결 조건 설정**

- 답변의 끝을 알리는 종결 문구를 명시적으로 제시할 수 있습니다.
- 이는 특히 긴 응답을 원할 때 유용합니다.

### **7. 다양한 프롬프트 스타일 시도**

- 질문, 명령, 이야기 시작 등 다양한 접근 방식을 시도하여 어떤 스타일이 가장 효과적인지 찾아봅니다.
- 다양한 접근 방식은 모델이 다른 관점에서 정보를 처리하는 데 도움이 될 수 있습니다.

### **8. 안전성 및 윤리적 고려**

- 모델이 부적절하거나 해로운 내용을 생성하지 않도록 주의합니다.
- 필요한 경우, 모델의 출력에 대한 후처리나 필터링을 고려합니다.

### **결론**

프롬프트 엔지니어링은 실험과 반복을 통해 발전합니다. 각 시도를 통해 배운 것을 적용하면서 점차적으로 원하는 결과에 더 가까워질 수 있습니다. 이러한 요령들은 모델을 더 효과적으로 활용하고, 원하는 정보를 얻는 데 도움이 될 것입니다.
"""

st.header("특정 목적을 가진 챗봇(데모)")
st.info("프롬프트 엔지니어링에 대해 알아볼 수 있는 Q&A 로봇입니다.")
st.error("프롬프트 엔지니어링에 대한 내용이 적용되어 있습니다.")

if "messages" not in st.session_state: # 딕셔너리 구조로 상태 저장
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 저는 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input(): # 채팅이 들어오면
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)  # streaming=True 실시간으로 답이 주르륵 나오는 것
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))