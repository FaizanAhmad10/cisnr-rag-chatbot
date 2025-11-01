import os
from dotenv import load_dotenv
from docx import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------- LOAD API KEY --------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------- EXTRACT TEXT FROM WORD FILE --------------------
def extract_text_from_docx(file_path):
    """Extract all text from a Word document."""
    try:
        doc = Document(file_path)
        full_text = []
        
        # Extract text from all paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                full_text.append(paragraph.text)
        
        # Also extract text from tables (if any)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        
        return '\n\n'.join(full_text)
    
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print(f"   Please make sure the file exists in: {os.path.abspath(file_path)}")
        exit(1)
    except Exception as e:
        print(f"❌ Error reading Word file: {e}")
        exit(1)

# -------------------- LOAD DATA FROM WORD FILE --------------------
print("📄 Loading data from Word file...")
word_file = 'cisnr_data.docx'  # Change this to your file name if different
cisnr_data = extract_text_from_docx(word_file)
print(f"✅ Loaded {len(cisnr_data)} characters from Word file")
print(f"   Preview: {cisnr_data[:100]}...\n")

# -------------------- TEXT SPLITTING --------------------
print("🔄 Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents([LangchainDocument(page_content=cisnr_data)])
print(f"✅ Created {len(docs)} text chunks\n")

# -------------------- EMBEDDINGS --------------------
print("🔄 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------- VECTOR STORE --------------------
print("🔄 Building vector store...")
persist_directory = "vector_store"
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("✅ Vector store created successfully\n")

# -------------------- GROQ LLM --------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

# -------------------- RAG CHAIN --------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
""")

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------- CHAT LOOP --------------------
print("=" * 60)
print("🤖 CISNR RAG Chatbot (Groq + ChromaDB)")
print("=" * 60)
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Goodbye!")
        break
    try:
        answer = rag_chain.invoke(query)
        print(f"Bot: {answer}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")





# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # -------------------- LOAD API KEY --------------------
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# # -------------------- CISNR DATASET --------------------
# cisnr_data = """
# CISNR (Centre of Intelligent System & Network Research) provides smart and intelligent solutions 
# to society that balance safety, economic growth, and the environment through accurate data 
# acquisition and technology. The organization focuses on product-oriented research targeting 
# safe city projects, which includes AI-based solutions for smart disaster management, smart 
# environment, and autonomous event detection.

# CISNR focuses on four main goals related to sustainable development:
# 1. Fostering National AI Capacity: Building national capacity to carry out R&D in Artificial Intelligence.
# 2. Empowering Innovators: Providing a platform and resources for researchers, developers, and engineers to innovate modern solutions.
# 3. Transforming the Future: Conducting research in AI to prevent or minimize natural and man-made disasters using data intelligence.
# 4. Unlocking Boundless Creativity: Developing innovative designs and intelligent systems.

# AquaCure: A water supply management system that provides real-time monitoring and control of water 
# quality and quantity. It addresses water protection and metering problems from water pumping tube 
# wells through storage tanks to consumers by providing real-time data acquisition and control of 
# water quality & quantification.

# ElectroCure: A low-cost smart metering solution using a meter-less architecture allowing a single 
# module to serve multiple consumers. It helps in detecting electricity theft and loss while providing 
# real-time data acquisition of electricity distribution networks.

# TransfoCure: An automatic system for transformer monitoring, control, data collection, and efficient 
# mechanisms to prevent errors and inefficiencies.

# AgroCure: A tech-based solution to reduce food insecurity, empower farmers by eliminating middlemen, 
# and provide a transparent platform for accessing real-time farm data and insights.

# MeteroCure: An image-based autonomous remote billing smart system to overcome underbilling issues. 
# It sends data through a cellular network to a central server, maintains complete past data history, 
# and records GPS coordinates of readings with date and time.

# CMS: A complaint management system for monitoring and tracking user complaints. It provides a complete 
# system that automatically handles CMS processes 24/7.

# Auto Filler: A system to prevent water spills and wastage using ultrasonic sensors, automatic switches, 
# and water flow sensors. It ensures undisturbed water supply, avoids spills, and is easy to install.
# """

# # -------------------- TEXT SPLITTING --------------------
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# docs = splitter.split_documents([Document(page_content=cisnr_data)])

# # -------------------- EMBEDDINGS --------------------
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # -------------------- VECTOR STORE --------------------
# persist_directory = "vector_store"
# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory=persist_directory
# )

# # -------------------- GROQ LLM (UPDATED MODEL) --------------------
# llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

# # -------------------- RAG CHAIN --------------------
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Create prompt template
# prompt = ChatPromptTemplate.from_template("""
# Answer the question based only on the following context:

# {context}

# Question: {question}

# Answer:
# """)

# # Helper function to format documents
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Build RAG chain using LCEL
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # -------------------- CHAT LOOP --------------------
# print("🤖 CISNR RAG Chatbot (Groq + ChromaDB)")
# print("Type 'exit' to quit.\n")

# while True:
#     query = input("You: ")
#     if query.lower() in ["exit", "quit"]:
#         break
#     try:
#         answer = rag_chain.invoke(query)
#         print(f"Bot: {answer}\n")
#     except Exception as e:
#         print(f"Error: {e}\n")