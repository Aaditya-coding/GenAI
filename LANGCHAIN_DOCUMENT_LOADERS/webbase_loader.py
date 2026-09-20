from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion",
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/Apple-MacBook-Laptop-18%E2%80%91core-20%E2%80%91core/dp/B0GR1LB81D/ref=sr_1_3?crid=1JMTHR0VXJ82V&dib=eyJ2IjoiMSJ9.BNwGwUU2hVsw2CAiCH9WcOrTbitMqW-NF15EWCczZQ_Ds9eVMUdo5mpDxrdLlW5zj6bOcE49IbWoHP7t7it--kMrUwhEsxYASsORfwb-VxtpAT4LQIRsLn9rSN4Ow2tX_aa_qASMnxvOqPqlhaJr5mweyYaA1zjq-XaMkMNWV5R3pCcNfycpv1CThZvogxUKfTx99xV6BvmSauujBxUjtvWglVM8QP8aq0ykEF0tXOc.YypkMRAd3Glj8kN9jIr7mZgTEj_UhmPQE8Q4n7yFaE4&dib_tag=se&keywords=macbook%2Bpro%2Bm5&qid=1786198581&sprefix=macbook%2Bpro%2Caps%2C383&sr=8-3&th=1'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the product that we are talking about?', 'text':docs[0].page_content}))