import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import TextLoader, JSONLoader, DirectoryLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

if __name__ == '__main__':


    #【1】建立文本的向量数据库
    file_dir = './math_knowledge'
    # 名称列表：解析几何、数分、高代、常微分方程
    subject_all = ['analytic_geometry','linear_algebra','mathematical_analysis','ordinary_differential_equation']
    retriever_all = {} #定义检索器存储字典，每个subject对应一个检索器

    #【1-1】载入数据load
    text_loader_kwargs = {'autodetect_encoding': True} #对应类型解码
    for subject in subject_all:
        path = f'{file_dir}/{subject}' #对应路径
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
        embedding_model = OpenAIEmbeddings()

        db = FAISS.from_documents(doc_list, embedding_model)
        retriever_name: str = f'{subject}' #定义本轮中检索器的名称
        retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3})  # 变成检索器,相关文档数k
        retriever_all[retriever_name] = retriever #存入retriever_all列表


    # 【2】建立查询链
    # 全局变量
    global rag_type
    global loop_max
    global loop
    loop_max = 5
    # 【2-1】prompt
    template1 = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    
    context: {context}
    
    Question: {question}
    
    """
    template2 = """Given a history of events and a question related to that history, 
    summarize the history and provide a simplified version of the question.
    Please do not give any answers to the questions, just make a summary of the historical questions.
    
    Question: {question}
    
    """
    template3 = """Please make the following judgments based on the questions given:
    If you are judged to be able to answer the question given, take care to be very sure that the answer can be combined with the actual situation and that no additional knowledge is required, then please write the final output at the end [Y].
    If you are not sure that the answer is correct, there are currently four kinds of knowledge, namely mathematical analysis, higher algebra, analytic geometry and ordinary differential equations, please determine which kind of knowledge the problem belongs to, if it is mathematical analysis, then the last part is added [mathematical_analysis], if it is linear algebra, Then add [linear_algebra] at the end, [ordinary_differential_equation] at the end if it belongs to ordinary differential equations, and [analytic_geometry] at the end if it belongs to analytic geometry.
    Attention, please! The answer must end with only one of the above.
    
    Question: {question}
    
    """

    prompt1 = ChatPromptTemplate.from_template(template1)
    prompt2 = ChatPromptTemplate.from_template(template2)
    prompt3 = ChatPromptTemplate.from_template(template3)
    # 【2-2】model
    # chat = QianfanChatEndpoint(model="ERNIE-Bot-4")
    chat = ChatOpenAI()


    # 【2-3】outputparser
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    class MathOutputParser(BaseOutputParser[list[str]]):
        """！！待修改：两个输出1输出答案2输出最后解析的YN等进行判断"""

        def parse(self, text: str) -> list[str]:
            """Parse the output of an LLM call"""
            return text.strip().split(",")  # 去除所有空格后用逗号分割成列表

    def outputparser3(st: str):
        result = re.search(r'\[.*?\]', st)
        return result


    # 【2-4】chain
    def judge_parallel(x):
        global rag_type, loop
        rag_type = x["type"]
        if rag_type != 'Y':
            loop += 1
            context = retriever_all[rag_type] | format_docs
        else:
            context = ' '
        return context

    #chain2
    rag_chain2 = (
        prompt2
        | chat
        | StrOutputParser() #得到提炼后的问题
        | RunnableParallel({"question2": RunnablePassthrough(), "type": prompt3 | chat | StrOutputParser | outputparser3})
        | judge_parallel
    )

    #chain1
    rag_chain1 = (
        RunnablePassthrough().assign(context = rag_chain2)
        | prompt1
        | chat
        | StrOutputParser(),
    )

    # 【3】运行
    go_on = True #控制可以多轮交互
    while go_on:
        query_text = input("你的问题：")
        #结束条件：exit
        if 'exit' in query_text:
            break
        print("AI需要回答的问题[{}]\n".format(query_text))

        #用户提问一次时
        res = rag_chain1.invoke({"question":query_text}) #注入问题和历史，返回回答
        print("本轮的思考结果为[{}]\n".format(res))
        query_text = res + query_text #将每轮循环的输出加入到下一轮的输入中
        loop += 1
        #结束条件1：chain思考了max次以上
        if loop > loop_max:
            break
        #结束条件2：Y和N代表没必要查询了，该公布答案或说不知道了
        if rag_type == 'Y' or rag_type == 'N':
            break
