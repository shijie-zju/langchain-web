import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import TextLoader, JSONLoader, DirectoryLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

if __name__ == '__main__':
    #【1】建立文本的向量数据库
    #【1-1】载入数据load
    path = "./knowledge"
    text_loader_kwargs = {'autodetect_encoding':True}
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs, show_progress=True, use_multithreading=True)
    docs = loader.load()
    pass

    #【1-2】分割数据split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
    )
    doc_list = [] #分割为Document对象并存入列表doc_list
    for doc in docs:
        tmp_docs = text_splitter.create_documents([doc.page_content])
        doc_list += tmp_docs

    #【1-3】文本向量化，存入数据库store
    # embedding_model = QianfanEmbeddingsEndpoint(model="bge_large_zh",
    #                                             endpoint="bge_large_zh")#更擅长中文的embedding
    embedding_model = OpenAIEmbeddings()

    # vectorstore = ElasticsearchStore(
    #     es_url=os.environ['ELASTIC_HOST_HTTP'],
    #     index_name="index_sd_1024_vectors",
    #     embedding=embedding_model,
    #     es_user="elastic",
    #     vector_query_field='question_vectors',
    #     es_password=os.environ['ELASTIC_ACCESS_PASSWORD']
    # )
    # #将本文加入数据库（仅运行一次，多次则会有冗余数据）
    # vectorstore.add_documents(doc_list)
    #得到一个向量化的【检索器】
    #retriever = vectorstore.as_retriever()

    db = FAISS.from_documents(doc_list, embedding_model)
    retriever = db.as_retriever(search_type = 'similarity', search_kwargs={"k":5})  # 变成检索器,相关文档数k

    #【2】建立查询链
    #【2-1】prompt
    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}
        
        Helpful answer:"""
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    #【2-2】model
    #chat = QianfanChatEndpoint(model="ERNIE-Bot-4")
    chat = ChatOpenAI()

    #【2-3】outputparser
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #【2-4】chain
    #检索+回答chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x:format(x["context"])))#不管传来的question
        | custom_rag_prompt
        | chat
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)#以“answer”：..格式加入并行中

    #【3】运行
    go_on = True
    while go_on:
        query_text = input("你的问题：")

        if 'exit' in query_text:
            break

        print("AI需要回答的问题[{}]\n".format(query_text))
        res = rag_chain_with_source.invoke(query_text)
        print(res)


