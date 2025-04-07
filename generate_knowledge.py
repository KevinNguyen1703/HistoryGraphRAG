import pandas as pd
from graphrag.custom_llm_call import call_openai

# df_naive_RAG = pd.read_excel("evaluation/validation-set-3/Naive-RAG.xlsx")
df_graph_RAG = pd.read_excel("evaluation/validation-set-4/graphrag-2.xlsx")

generation_question_prommpt = """
Bạn là một trợ lí để giúp tôi viết các bài chia sẻ trên wikipedia để trả lời các câu hỏi trắc nghiệm lịch sử. Tôi sẽ cung cấp các câu hỏi trắc nghiệm và đáp án, bạn hãy giúp tôi tạo ra các đoạn văn bản giải thích cho lựa chọn trên. Yêu cầu:
- Văn bản có thể thêm các chủ thể liên quan đến câu hỏi
- Văn bản cần mang tính chi tiết về cả những chủ thể có trong câu hỏi và những chủ thể liên quan
- Đầu ra chỉ cần các đoạn văn bản liên quan, không cần các yêu tố suy lận hoặc dẫn nhập.
Đầu vào:
{question}
Câu trả lời là: {answer}
"""
output_folder = 'dataset/text-extracted/addition_knowledge'
for i in range(len(df_graph_RAG)):
    if  df_graph_RAG['Answer'][i] != df_graph_RAG['LLM Answer'][i]:
        response = call_openai(generation_question_prommpt.format(question=df_graph_RAG['Question'][i],answer=df_graph_RAG['Answer'][i]))
        output_file = f"{output_folder}/cau_{i+1}.txt"
        with open(output_file,"w") as file:
            file.write(response)
