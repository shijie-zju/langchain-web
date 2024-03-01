from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseOutputParser

#解析器可以自己定义，也可以用现成的
class MathOutputParser(BaseOutputParser[list[str]]):
    """Parse the output of ...to..."""

    def parse(self, text:str) -> list[str]:
        """Parse the output of an LLM call"""
        return text.strip().split(",") #去除所有空格后用逗号分割成列表

if __name__ == '__main__':

    template = """You are a helpful assistant who knows to solve math problems.
    Try to think step by step."""
    human_template = "The question is {query}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("user", human_template)
    ])

    llm = ChatOpenAI()

    output_parser = StrOutputParser()
    #output_parser = MathOutputParser()

    chain = prompt | llm | output_parser

    res = chain.invoke({"query": "What is the Hydrodynamic equation?"})
    print(res)