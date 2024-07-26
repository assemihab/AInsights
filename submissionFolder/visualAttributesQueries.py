
################imports################
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_openai import ChatOpenAI 
from langchain.output_parsers import ResponseSchema, StructuredOutputParser,PydanticToolsParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import sqlite3
from langchain.schema import HumanMessage, SystemMessage, AIMessage 

#################LLM################
load_dotenv()
API_KEY = os.getenv("API_KEY")
nlpfpath='Intern NLP Dataset.xlsx'
os.environ["OPENAI_API_KEY"] = API_KEY
# modell="gpt-3.5-turbo"
modell="gpt-4o"
llm = ChatOpenAI(model=modell, temperature=0) 
#################Data################
def create_db(nlpfpath,dbfileName):
    df=pd.read_excel(nlpfpath)
    # columns=df.columns.tolist()
    sample=df.head().to_string()
    df=pd.read_excel(nlpfpath)
    conn = sqlite3.connect(dbfileName)
    c = conn.cursor()
    df.to_sql('nlp', conn, if_exists='replace', index = False)
    conn.commit()
    conn.close()
    return sample,dbfileName
sample,dbfilename=create_db(nlpfpath,'nlp.db')
def read_sql_query(query,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(query)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    return rows
################Prompt templates################

#generating subqueries#
class SubQuery(BaseModel): 
    """Search over a database of Visual Attribute Time Series Dataset""" 
    sub_query: str = Field(description="A very specific query against the database.", 
    )

system = f"""You are an expert at converting user questions into database 
queries. \ 
You have access to a database of Visual Attribute Time Series Dataset .  \ 
Perform query decomposition. Given a user question, break it down into 
distinct sub questions that \ 
you need to answer in order to answer the original question. 
If there are acronyms or words you are not familiar with, do not try to 
rephrase them.""" 
prompt = ChatPromptTemplate.from_messages( 
[ 
("system", system), 
("human", "{question}"), 
] 
) 
llm_with_tools = llm.bind_tools([SubQuery]) 
parser = PydanticToolsParser(tools=[SubQuery]) 
query_analyzer = prompt | llm_with_tools | parser 



#define the output parser response schemas#
response_schemas = [
    ResponseSchema(name="query", description="structured query to be executed against the sql lite3."),
    ResponseSchema(
        name="description",
        description="the description of the result the names of selected columns",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
template = PromptTemplate(
    template="answer the users question: {question}\n as database query on this database with this head sample {dataset} and table name nlp: .\n{format_instructions}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions,"dataset":sample},
)
useroutputtemplate = PromptTemplate(
    template="The answer for this {description} is: {query} make it in presentable format",
    input_variables=["description", "query"],
)
presentingchain=useroutputtemplate|llm
#################main################
def process_query(question, dbfilename, query_analyzer, template, llm, output_parser):
    subqueries = query_analyzer.invoke({"question": question})
    # print("The subqueries are:", subqueries)
    
    base_conversation = [
        SystemMessage(content="You are an expert at converting user questions into database queries")
    ]
    
    for subquery in subqueries:
        human_format = template.format(question=subquery)
        base_conversation.append(HumanMessage(content=human_format))
        
        ai_response = llm.invoke(base_conversation)
        base_conversation.append(AIMessage(content=ai_response.content))
        
        parsed_output = output_parser.parse(ai_response.content)
        query = parsed_output['query']
        
        success = False
        for _ in range(10):
            try:
                rows = read_sql_query(query, dbfilename)
                results=presentingchain.invoke({"description":parsed_output['description'],"query":rows})

                print(results.content)
                success = True
                break
            except Exception as e:
                print(f"Query failed with error: {e}. Retrying...")
                human_error = f"The query {query} failed with error {e}. Rewrite it."
                base_conversation.append(HumanMessage(content=human_error))
                
                ai_response = llm.invoke(base_conversation)
                base_conversation.append(AIMessage(content=ai_response.content))
                
                parsed_output = output_parser.parse(ai_response.content)
                query = parsed_output['query']
        
        if not success:
            print("The query failed after 10 retries. Please rewrite your prompt.")

def main(nlpfpath, dbfilename, query_analyzer, template, llm, output_parser):
    sample, dbfilename = create_db(nlpfpath, dbfilename)
    print("Here is a sample from the dataset:")
    print(sample)
    
    while True:
        question = input("Enter the question you want to ask the database: ")
        process_query(question, dbfilename, query_analyzer, template, llm, output_parser)
        
        inputt = input("Write 'q' to quit or cls to clear and continue: ")
        if inputt.lower() == 'q':
            break
        elif inputt.lower() == 'cls':
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            print("Reenter the question or rewrite it in a different format.")

if __name__ == '__main__':
    main(nlpfpath, 'nlp.db', query_analyzer, template, llm, output_parser)



