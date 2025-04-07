from graphrag.custom_llm_call import ollama_complete, gpt_4o_mini_complete, gpt_4o_complete, call_openai
from graphrag.prompt import PROMPTS
import os
from tqdm import tqdm
import json
import uuid
from utils import add_doc_id, load_json, save_json
def read_files_in_order(folder):
    files = sorted(os.listdir(folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    texts = []
    for file in files:
        if file.endswith(".txt"):  # Chỉ xử lý file văn bản
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
    return texts

def create_batches(texts, chunk_size, overlap):
    batches = []
    num_pages = len(texts)
    for start in range(0, num_pages, chunk_size - overlap):
        end = min(start + chunk_size, num_pages)
        batch_text = "\n".join(texts[start:end])
        batches.append(batch_text)
    return batches


def clean_dataset(ocr_folder, output_dir="dataset/process_chunk.json" , chunk_size=10, overlap=2):
    texts = read_files_in_order(ocr_folder)
    text_batches = create_batches(texts, chunk_size, overlap)
    processed_chunks = []
    PROMPT_TEMPLATE=PROMPTS['process_chunk']

    for batch in tqdm(text_batches, desc="Processing clean data process"):
        response = call_openai(PROMPT_TEMPLATE.format(input_text=batch))
        chunks = response.split("<|>")  # Tách kết quả theo định dạng quy ước
        processed_chunks.extend(chunks)

    save_json(processed_chunks,output_dir)
    print(f"✅ Xử lý hoàn tất! Kết quả lưu tại: {ocr_folder}")

def split_into_chunks(docs, word_limit=600):
    """
    Split documents into chunks where each chunk has around 600 words.
    Uses sentences as delimiters to keep chunks meaningful.
    """
    chunks = []

    for doc in docs:
        doc_id = doc["doc_id"]
        sentences = doc["content"].split(".")
        chunk = []
        chunk_word_count = 0

        for sentence in sentences:
            words = sentence.strip().split()
            word_count = len(words)

            if chunk_word_count + word_count > word_limit and chunk:
                # Save the current chunk and start a new one
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "content": " ".join(chunk).strip()
                })
                chunk = []
                chunk_word_count = 0

            chunk.append(sentence.strip())
            chunk_word_count += word_count

        # Add the last chunk if it contains any 10-old
        if chunk:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "content": " ".join(chunk).strip()
            })

    return chunks


if __name__ == "__main__":
    ocr_folder="dataset/text-extracted/10-chan-troi-sang-tao-manual"
    processed_chunk_file = "../dataset/chunk/10-chan-troi-sang-tao/process_chunk.json"  # Input JSON file
    output_file_docs = "../dataset/chunk/10-chan-troi-sang-tao/documents_with_id.json"  # Output file with doc_id
    output_file_chunks = "../dataset/chunk/10-chan-troi-sang-tao/chunks.json"  # Output file with document chunks

    # Load data
    clean_dataset(ocr_folder=ocr_folder,output_dir=processed_chunk_file)
    documents = load_json(processed_chunk_file)

    # Process documents
    documents_with_id = add_doc_id(documents)
    chunks = split_into_chunks(documents_with_id)

    # Save outputs
    save_json(documents_with_id, output_file_docs)
    save_json(chunks, output_file_chunks)
