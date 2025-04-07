from graphrag.custom_llm_call import call_openai

split_summary_history_doc_prompt="""
Bạn là một chuyên gia về lịch sử có thể tạo ra các đoạn ngữ liệu văn bản cho quá trình huấn luyện mô hình ngôn ngữ. 
Hãy giúp tôi chỉnh sửa các lỗi chính tả có trong bài tóm tắt lịch sử trên và chia nó thành các đoạn văn đầy đủ ý nghĩa.
Bạn có thể thêm các nội dung khác hoặc loại bỏ các đoạn dư thừa nếu cần thiết.
Đây sẽ là ngữ liệu văn bản cho quá trình huấn luyện một mô hình ngôn ngữ khác nên cần giải thích chi tiết và thêm các nội dung bổ sung cho từng chủ đề hoặc các đối tượng.
Định dạng đầu ra:
<Đoạn văn 1><|><Đoạn văn 2><|>Đoạn văn 3<|>...
Đầu vào:
{content}
"""

def correct_and_split(file_path):
    text = ""
    with open(file_path,"r") as f:
        text = f.read()
    text_list = text.split("CHƢƠNG")[1:]
    for idx, text_chunk in enumerate(text_list):
        response = call_openai(split_summary_history_doc_prompt.format(content=text_chunk))

        file = open(f"dataset/text-extracted/tomtatlichsu12/chuong_{idx+1}.txt", "w")
        file.write(response)
        file.close()

if __name__ == "__main__":
    correct_and_split("../dataset/text-extracted/tomtatlichsu12/full.txt")
