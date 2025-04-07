import os
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from graphrag.data_chunk import run_chunk
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import openai
import requests
import json

sum_prompt="""
Generate a structured summary from the provided historical source (report, article, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

EVENT: Đề cập đến các sự kiện lịch sử quan trọng liên quan đến văn minh Văn Lang - Âu Lạc.
PERSONALITIES: Liệt kê các nhân vật lịch sử quan trọng trong văn minh Văn Lang - Âu Lạc (ví dụ: An Dương Vương, Lạc Long Quân, Âu Cơ).
DATES: Bao gồm các ngày tháng hoặc thời kỳ quan trọng liên quan đến văn minh Văn Lang - Âu Lạc (ví dụ: khoảng thế kỉ VII TCN - 179 TCN).
LOCATIONS: Liệt kê các địa danh quan trọng trong văn minh Văn Lang - Âu Lạc (ví dụ: Phong Châu, Cổ Loa, Bắc Bộ, Bắc Trung Bộ).
SOCIAL_CONTEXT: Phân tích các yếu tố xã hội và cấu trúc cộng đồng trong văn minh Văn Lang - Âu Lạc (ví dụ: các nhóm tộc người, cộng đồng người Việt cổ).
ECONOMIC_CONTEXT: Các hoạt động kinh tế, như nông nghiệp, thủ công nghiệp, và buôn bán trong văn minh Văn Lang - Âu Lạc (ví dụ: đúc đồng, trồng lúa nước).
CULTURAL_CONTEXT: Các yếu tố văn hóa và tín ngưỡng trong văn minh Văn Lang - Âu Lạc (ví dụ: tín ngưỡng thờ Mặt Trời, văn học truyền miệng, lễ hội).
TECHNOLOGICAL_ADVANCEMENTS: Các thành tựu khoa học, kỹ thuật trong văn minh Văn Lang - Âu Lạc (ví dụ: lưỡi cày đồng, trống đồng Đông Sơn).
MATERIAL_LIFE: Các khía cạnh đời sống vật chất, như ăn, mặc, ở và phương tiện đi lại (ví dụ: nhà sàn, trang phục, phương tiện giao thông).
SPIRITUAL_LIFE: Các khía cạnh đời sống tinh thần, bao gồm tín ngưỡng, văn học và các phong tục (ví dụ: thờ cúng tổ tiên, các câu chuyện thần thoại).
IMPACT: Tác động của văn minh Văn Lang - Âu Lạc đối với lịch sử và nền văn hóa Việt Nam.

Each category should be addressed only if relevant to the content of the historical source. Ensure the summary is clear and direct, suitable for quick reference.
"""
def get_embedding(text, mod = "text-embedding-3-small"):
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        input=text,
        model=mod
    )

    return response.data[0].embedding

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element


def add_sum(n4j, content, gid):
    sum = process_chunks(content)
    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})

    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    return s

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        # chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
        chunks.append(encoding.decode(words[i:i + tokens]))
    return chunks
def call_openai_api(chunk):
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content

def process_chunks(content):
    chunks = split_into_chunks(content)
    print(chunks)
    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))
    # print(responses)
    return responses

def creat_metagraph(content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content


    # content = run_chunk(content)

    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    add_sum(n4j, whole_chunk, gid)
    return n4j

sys_p = """
Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
"""
def seq_ret(n4j, sumq):
    rating_list = []
    sumk = []
    gids = []
    sum_query = """
        MATCH (s:Summary)
        RETURN s.content, s.gid
        """
    res = n4j.query(sum_query)
    for r in res:
        sumk.append(r['s.content'])
        gids.append(r['s.gid'])

    for sk in sumk:
        sk = sk[0]
        rate = call_llm(sys_p, "The two summaries for comparison are: \n Summary 1: " + sk + "\n Summary 2: " + sumq[0])
        if "totally not similar" in rate:
            rating_list.append(0)
        elif "not similar" in rate:
            rating_list.append(1)
        elif "general" in rate:
            rating_list.append(2)
        elif "very similar" in rate:
            rating_list.append(4)
        elif "similar" in rate:
            rating_list.append(3)
        else:
            print("llm returns no relevant rate")
            rating_list.append(-1)

    ind = find_index_of_largest(rating_list)
    # print('ind is', ind)

    gid = gids[ind]

    return gid
def call_llm(sys, user):
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f" {user}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content

def call_ollama(
    prompt, system_prompt=None, history_messages=[], model="llama3.1:8b", **kwargs
) -> str:
    """
    Generates a response using the Ollama model, similar to `gpt_4o_mini_complete`.

    :param prompt: The user input prompt.
    :param system_prompt: An optional system message to guide the response.
    :param history_messages: A list of past messages for context.
    :param model: The name of the Ollama model (default: "mistral").
    :param kwargs: Additional parameters.
    :return: The generated response as a string.
    """
    url = "http://localhost:11434/api/generate"

    # Constructing the complete context with system prompt and history
    context = []
    if system_prompt:
        context.append(system_prompt)
    context.extend(history_messages)

    payload = {
        "model": model,  # Default is "mistral"
        "prompt": prompt,
        "context": context,
        "stream": False,  # Set to True if you want a streaming response
    }
    payload.update(kwargs)  # Merge additional parameters

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    # Making the request
    response_text = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8'))
                response_text += data.get("response", "")
                if data.get("done", False):
                    break  # Stop when "done": true is encountered
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return response_text

def find_index_of_largest(nums):
    # Sorting the list while keeping track of the original indexes
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))

    # Extracting the original index of the largest element
    largest_original_index = sorted_with_index[-1][1]

    return largest_original_index

def get_response(n4j, gid, query):
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    user_one = query + "Thông tin bổ sung bao gồm:" +  "".join(selfcont)
    res = call_ollama(user_one, system_prompt=sys_prompt_one)
    user_two = query + "\nCâu trả lời trước là:" + res + "\nThông tin bổ sung bao gồm:" +  "".join(linkcont)
    res = call_llm(sys_prompt_two,user_two)
    return res

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary // Ensure each pair is processed once and exclude "Summary" nodes in relationships
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def link_context(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

sys_prompt_one = """
Vui lòng trả lời câu hỏi bằng cách sử dụng thông tin chi tiết được hỗ trợ bởi dữ liệu đồ thị có liên quan đến thông tin lịch sử.
"""

sys_prompt_two = """
Sửa đổi câu trả lời cho câu hỏi bằng cách sử dụng các tài liệu tham khảo được cung cấp. Bao gồm các trích dẫn chính xác có liên quan đến câu trả lời của bạn. Bạn có thể sử dụng nhiều trích dẫn cùng lúc, biểu thị mỗi trích dẫn bằng số chỉ mục tham chiếu. Ví dụ, trích dẫn tài liệu đầu tiên và thứ ba là [1][3]. Nếu các tài liệu tham khảo không liên quan đến câu trả lời, chỉ cần cung cấp câu trả lời ngắn gọn cho câu hỏi ban đầu."""