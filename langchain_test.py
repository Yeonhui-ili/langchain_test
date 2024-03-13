from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# api key값 가져오기 (보안)
load_dotenv()
my_openai_api_key=os.getenv("OPENAI_API_KEY")
# 모델 초기화
llm = ChatOpenAI(openai_api_key = my_openai_api_key)

#######################################################################

# langchain 모델 기본 사용
output1 = llm.invoke("2024년 청년 지원 정책에 대하여 알려줘")
print(output1.content)

#######################################################################

# Template 기반 사용
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야"),
        ("user", "{input}")
    ]
)

# pipe 연산자 : (A | B) A객체나 값을 B 함수에 파이프로 전달. 둘이 연결됨. 묶어서 실행가능
chain = prompt | llm

output2 = chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘"})
print(output2.content)

#######################################################################

# 파싱하기
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# 값 탬플릿, 모델, 그중 아웃풋만 골라서 
chain = prompt | llm | output_parser

output3 = chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘"})
print(output3) # 위와 달리 따로 지정하지 않아도 content부분만 나옴

#######################################################################

# pip install beautifulsoup4 크롤링 라이브러리
# pip install faiss-cpu

# 검색 기능 적용
# 웹데이터를 크롤링해서 특정 도메인에 최적화된 서비스 구현 등에 이용
# 의미있는 내용만 추출해서 보내겠다. 검색 기반 라이브러리 사용해서 관심부분 위주의 대답 생성
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings # 임베딩 : 자연어나 이미지 등을 벡터로 전환
from langchain_community.vectorstores import FAISS # FAISS 백터 유사성 검색
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

# 웹페이지 크롤링
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")
docs = loader.load()
# 임베딩 객체 생성
embeddings = OpenAIEmbeddings(openai_api_key= my_openai_api_key)
# 택스트 스플리터 객체 생성, 문서 처리 후 저장
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 문서 백터화
vector = FAISS.from_documents(documents, embeddings)


# 크롤링한 사이트 기반으로 응답
prompt = ChatPromptTemplate.from_template("""제공된 context만으로 기반으로 아래의 질문에 대답하기
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# 바로 Docs 내용을 반영하는 것도 가능
output4 = document_chain.invoke({
    "input": "국민취업지원제도가 뭐야",
    "context": [Document(page_content="""국민취업지원제도란?

취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
[출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24""")]
})
print(output4)

# 크롤링으로 가져온 벡터값 이용
retriever = vector.as_retriever() #context에 들어갈 크롤링해온 내용
retrieval_chain = create_retrieval_chain(retriever, document_chain)
output5 = retrieval_chain.invoke({"input": "국민취업지원제도가 뭐야"})
print(output5["answer"])

