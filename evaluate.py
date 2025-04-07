from graphrag.custom_llm_call import call_openai, ollama_complete
import pandas as pd
from graphrag.prompt import PROMPTS
from test_RAG import graph_rag
from langchain_core.vectorstores import VectorStoreRetriever
from openai import OpenAI
LLM_PROMPT_TEMPLATE = PROMPTS["validation"]

def process_questions(input_file: str, output_file: str, debug = False, model = None, retriever: VectorStoreRetriever = None) -> None:
    df = pd.read_excel(input_file)

    llm_answers = []
    validations = []
    correct_count = 0
    total_questions = len(df)
    for index, row in df.iterrows():
        question = row["Question"]
        correct_answer = row["Answer"].strip()

        query = LLM_PROMPT_TEMPLATE.format(question=question)
        if retriever:
            contexts = retriever.invoke(question)
            query += "\n Dưới đây là các thông tin bổ sung: "
            for idx, context in enumerate(contexts):
                query += f"\n{idx}: {context.page_content}"

        if model == 'gpt':
            response = call_openai(query)
        else:
            response = ollama_complete(query, model=model)

        llm_answers.append(response)

        if debug:
            print(f"Query: \n {query}\n ---> Answer: {response} \n-----------------------\n")
        if response == correct_answer:
            validation = "✅"
            correct_count += 1
        else:
            validation = "❌"

        validations.append(validation)

    df["LLM Answer"] = llm_answers
    df["Validation"] = validations

    df.to_excel(output_file, index=False)

    print(f"Kết quả đã được lưu vào {output_file}")
    print(f"Số câu trả lời đúng: {correct_count}/{total_questions}")


