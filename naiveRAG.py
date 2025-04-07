import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from graphrag.customllm import CustomLlama
from pymilvus import FieldSchema, Collection, CollectionSchema, DataType, connections
from pymilvus.exceptions import MilvusException
from openai import OpenAI
import re
from tqdm import tqdm
from evaluate import process_questions
client = OpenAI()
COLLECTION = "lich_su_12"
EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-ada-002")
class SectionTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str):
        # Split the text into sections based on "I.", "II.", "1.", "2."
        section_pattern = re.compile(r"([IVXLCDM0-9]+\.)[\s]*([A-Za-z0-9À-ỹ\W]+)")
        sections = re.split(section_pattern, text)

        # Combine section headings with their content
        structured_sections = []
        for i in range(1, len(sections), 3):
            heading = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            structured_sections.append(f"{heading} {content}")

        # Now split each section into chunks
        chunks = []
        for section in structured_sections:
            start = 0
            while start < len(section):
                end = start + self.chunk_size
                chunk = section[start:end]
                chunks.append(chunk)
                start = end - self.chunk_overlap  # overlap with previous chunk
        return chunks


def load_and_split_corpus(folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                corpus.append(content)

    # Initialize the splitter
    text_splitter = SectionTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    # Split the text into chunks based on sections and subsections
    for document in corpus:
        document_chunks = text_splitter.split_text(document)
        chunks.extend(document_chunks)

    return chunks


def init_milvus_client(collection_name):
    # Kết nối đến Milvus
    connections.connect("default", host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'))
    print("Kết nối đến Milvus thành công.")

    MAX_CONTENT_LENGTH = 4096
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="plain_text", dtype=DataType.VARCHAR, max_length=MAX_CONTENT_LENGTH)
    ]
    collection_schema = CollectionSchema(fields)

    # Create or connect to the collection
    try:
        # Check if collection exists, otherwise create it
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' đã tồn tại.")
    except Exception as e:
        # Collection doesn't exist, create new one
        collection = Collection(name=collection_name, schema=collection_schema)
        print(f"Collection '{collection_name}' đã được tạo.")

    return collection_name


def get_embeddings(chunk, embedded_model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=embedded_model,
        input=chunk
    )
    embedding = response.data[0].embedding  # Truy cập đúng vào dữ liệu embedding
    return embedding

# Function to insert embeddings into Milvus collection
def insert_embeddings_into_milvus(collection_name, chunks):
    # Connect to Milvus
    connections.connect("default", host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'))
    print("Connected to Milvus.")

    # Create Milvus schema
    MAX_CONTENT_LENGTH = 4096
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="plain_text", dtype=DataType.VARCHAR, max_length=MAX_CONTENT_LENGTH)
    ]
    collection_schema = CollectionSchema(fields)

    try:
        # Check if collection exists, otherwise create it
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except MilvusException:
        # Collection doesn't exist, create new one
        collection = Collection(name=collection_name, schema=collection_schema)
        print(f"Collection '{collection_name}' created.")

    for idx, chunk in tqdm(enumerate(chunks), desc="Embedding chunk: "):
        embedding = get_embeddings(chunk)
        if not embedding:
            print("Failed to generate embeddings.")
            return
        # Insert data into the collection
        collection.insert([[str(idx),], [embedding,], [chunk,]])
        print(f"Inserted {idx}-th chunks into Milvus collection '{collection_name}'.")
def get_milvus_client():
    # Connect to the local Milvus server
    # connections.connect("default", host="localhost", port="19530")
    connections.connect("default", host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'))

    search_params = {
        "metric_type": "L2",  # Hoặc "IP" tùy vào loại metric bạn dùng
        "params": {"nprobe": 10}
    }
    return Milvus(
        collection_name=COLLECTION,
        embedding_function=EMBEDDING_FUNCTION,
        connection_args={"host": os.getenv('MILVUS_HOST'), "port": os.getenv('MILVUS_PORT')},
        primary_field="id",
        vector_field="vector",
        text_field="plain_text",
        search_params=search_params
    )

def chunking_corpus(folder_path: str = "dataset/text-extracted/10-chan-troi-sang-tao-manual", collection_name: str ="lich_su_12"):
    chunks = load_and_split_corpus(folder_path)
    init_milvus_client(collection_name=collection_name)
    insert_embeddings_into_milvus(collection_name=collection_name, chunks=chunks)


if __name__ == "__main__":
    vector_db: Milvus = get_milvus_client()
    milvus_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    input_file = "evaluation/validation-4.xlsx"
    output_file = "evaluation/validation-set-4/GPT-RAG-5-retriver.xlsx"
    process_questions(input_file,output_file,model='gpt',retriever=milvus_retriever)
