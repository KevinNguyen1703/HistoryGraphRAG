import torch
import openai
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from graphrag.custom_llm_call import call_openai, ollama_complete
import ast# 🔹 Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gumiho123"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 🔹 Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# 🔹 Neo4j Connection
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# 🔹 Function to extract entities from a user query using OpenAI
def extract_entities_from_query(query):
    system_prompt = """
Bạn là chuyên gia trong lĩnh vực lịch sử. Xác định các thực thể lịch sử từ câu hỏi và các đáp án trong truy vấn đã cho. Danh sách các thực thể có thể nằm trong câu hỏi hoặc từ các đáp án được cung cấp.

Định dạng phản hồi như sau:
    ["Entity1", "Entity2", "Entity3"]

Câu hỏi cần trích xuất:
{question}
    """

    response = call_openai(system_prompt.format(question=query))
    print(response)
    return ast.literal_eval(response) # Convert string JSON to Python dict

# 🔹 Function to get query embedding
def get_query_embedding(query):
    return embedding_model.encode(query).tolist()


# 🔹 Function to retrieve relevant nodes & relationships from Neo4j based on extracted entities
def retrieve_graph_context(entities, top_k=5):
    retrieved_nodes = []
    related_info = []

    for entity in entities:
        # Retrieve entity descriptions
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $entity_name
        RETURN e.name AS name, e.description AS description, gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """

        entity_embedding = get_query_embedding(entity)
        with driver.session() as session:
            results = session.run(cypher_query, entity_name=entity, query_embedding=entity_embedding, top_k=top_k)
            retrieved_nodes.extend([{"name": row["name"], "description": row["description"]} for row in results])

        # Retrieve relationships of the found entities
        cypher_rel_query = """
        MATCH (a:Entity {name: $entity_name})-[r:RELATED_TO]->(b:Entity)
        RETURN a.name AS source, b.name AS target, r.description AS relation_desc
        """
        with driver.session() as session:
            results = session.run(cypher_rel_query, entity_name=entity)
            for row in results:
                related_info.append({
                    "source": row["source"],
                    "target": row["target"],
                    "relation_desc": row["relation_desc"]
                })

    return retrieved_nodes, related_info


# 🔹 Function to format retrieved graph data as context
def format_context(retrieved_nodes, relationships):
    context = "Dưới đây là một số bối cảnh lịch sử có liên quan:\n\n"

    for node in retrieved_nodes:
        context += f"- {node['name']}: {node['description']}\n"

    if relationships:
        context += "\nRelationships:\n"
        for rel in relationships:
            context += f"- {rel['source']} → {rel['target']}: {rel['relation_desc']}\n"

    return context


# 🔹 Function to generate response using Mistral
def generate_response(query, context, model):
    prompt = f"{query}\n\n{context}"

    response = ollama_complete(prompt, model=model)
    return response

def graph_rag(query, model="deepseek-v2:16b"):
    print("🔎 Extracting entities from query using OpenAI...")
    try:
        extracted_entities = extract_entities_from_query(query)
    except:
        extracted_entities = []
    if not extracted_entities:
        return "⚠️ No relevant entities extracted from the query."

    print(f"🗂 Identified entities: {extracted_entities}")

    print("🔍 Retrieving relevant graph context from Neo4j...")
    retrieved_nodes, relationships = retrieve_graph_context(extracted_entities)

    if not retrieved_nodes:
        return "⚠️ No relevant historical context found in the graph database."

    print("📖 Formatting context...")
    context = format_context(retrieved_nodes, relationships)

    print("🤖 Generating response using Mistral...")
    response = generate_response(query, context, model)

    return response


# 🔹 Example Usage
if __name__ == "__main__":
    user_query= """
Bạn là một trợ lý AI chuyên gia về trả lời câu hỏi trắc nghiệm. Hãy đọc kỹ câu hỏi sau và chỉ trả lời bằng một ký tự đại diện cho đáp án đúng (A, B, C, D). Không giải thích, không đưa thêm thông tin, chỉ trả về một ký tự duy nhất:
Câu hỏi:

Thể loại văn học nào của Trung Quốc thời cổ - trung đại đã đạt đến đỉnh cao của nghệ thuật?
A. Sử thi. 
B. Thơ Đường. 
C. Truyện ngắn. 
D. Kịch.

Định dạng đầu ra:
- Chỉ trả về một ký tự (A, B, C hoặc D)

    """
    response = graph_rag(user_query)
    print("\n💡 AI Response:\n", response)

# Close Neo4j Connection
driver.close()
