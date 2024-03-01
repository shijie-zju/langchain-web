from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def format_docs(docs):
    """将检索的各项内容进行分段"""
    return "\n\n".join([d.page_content for d in docs])

if __name__ == '__main__':

    loader = TextLoader("knowledge2/ziranbianzhengfa.txt", autodetect_encoding=True)
    documents = loader.load() #特定格式文件的加载器
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #分块的大小，覆盖的多少
    texts = text_splitter.split_documents(documents) #分割成块
    embeddings = OpenAIEmbeddings() #文本转为高维向量
    db = FAISS.from_documents(texts, embeddings) #存储至faiss数据库
    retriever = db.as_retriever() #变成检索器
    #docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson") #已具备检索功能

    template = """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    res = chain.invoke("什么是自然辨证法的重要理论基础?")
    print(res)