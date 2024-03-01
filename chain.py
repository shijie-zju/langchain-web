# Requires:
# pip install langchain docarray tiktoken

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

if __name__ == '__main__':

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever() #构建数据库的检索对象

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    output_parser = StrOutputParser()

    """RunnableParallel可理解为一个可以invoke的对象，优势在于并行地将context和question两个结果同时传给prompt"""
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()} #question这里RunnablePassthrough()是接收所传入的参数内容
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    chain.invoke("where did harrison work?")