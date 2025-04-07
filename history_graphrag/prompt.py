GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "extract_entity"
] = """
-Nhiệm vụ-
Bạn là trợ lý thông minh hỗ trợ nhà nghiên cứu lịch sử phân tích các tuyên bố liên quan đến các thực thể lịch sử trong một văn bản.

-Mục tiêu-
Trích xuất các thực thể phù hợp với danh sách cho trước và xác định tất cả tuyên bố liên quan đến những thực thể này theo mô tả đã đề ra.

-Các bước-

Xác định tất cả thực thể lịch sử khớp với danh sách thực thể (có thể là tên cụ thể hoặc loại thực thể như nhân vật, triều đại, sự kiện…).
Với mỗi thực thể, trích xuất các tuyên bố liên quan, bao gồm:
Chủ thể: Thực thể thực hiện hành động trong tuyên bố.
Đối tượng: Thực thể liên quan hoặc bị ảnh hưởng, nếu không có thì ghi KHÔNG XÁC ĐỊNH.
Loại tuyên bố: Danh mục chính xác của tuyên bố.
Trạng thái: ĐÚNG, SAI, hoặc NGHI VẤN.
Mô tả: Giải thích chi tiết kèm bằng chứng.
Thời gian: Khoảng thời gian tuyên bố được đưa ra, định dạng ISO-8601, nếu không rõ thì ghi KHÔNG XÁC ĐỊNH.
Nguồn: Tất cả các trích dẫn từ văn bản gốc liên quan đến tuyên bố.
Mỗi tuyên bố được định dạng:
(<chủ_thể>{dấu_phân_tách}<đối_tượng>{dấu_phân_tách}<loại_tuyên_bố>{dấu_phân_tách}<trạng_thái>{dấu_phân_tách}<ngày_bắt_đầu>{dấu_phân_tách}<ngày_kết_thúc>{dấu_phân_tách}<mô_tả>{dấu_phân_tách}<nguồn>)

Trả về danh sách các tuyên bố, sử dụng {dấu_phân_tách_bản_ghi} giữa các bản ghi và kết thúc bằng {dấu_kết_thúc}.
Văn bản đầu vào:
"""
PROMPTS[
    "process_chunk"
] = """
Bạn là một chuyên gia xử lý ngôn ngữ tự nhiên (NLP) với nhiệm vụ xử lý văn bản trích xuất từ OCR của sách lịch sử. Hãy thực hiện các bước sau:

1. Chỉnh sửa lỗi OCR: Sửa lỗi chính tả, lỗi nhận diện ký tự, lỗi định dạng, lỗi dấu câu và lỗi ngữ pháp.
2. Loại bỏ thông tin không cần thiết: Bỏ tiêu đề, thông tin nhà xuất bản, biên soạn, mục lục, số trang, câu hỏi học sinh, bản quyền và phần không liên quan.
3. Chia văn bản thành các đoạn hợp lý (Chunking):
- Mỗi đoạn khoảng 600 tokens, có 100 tokens overlap giữa các đoạn.
- Giữ ngữ cảnh liền mạch, không cắt rời ý quan trọng.
4. Nếu có các câu hỏi trong bài viết, hãy trả lời câu hỏi đó và thêm nó vào nội dung của đoạn văn bản.
5. Định dạng đầu ra: Trả về kết quả dưới dạng <chunk1><|><chunk2><|><chunk3> mà không có bất kỳ giải thích nào.
Dưới đây là văn bản cần xử lý:
Văn bản cần xử lý:
```
{input_text}
```
"""

PROMPTS[
    "extract_entity"
] = """
Mục tiêu  
Xác định các thực thể lịch sử từ một văn bản đầu vào và mối quan hệ giữa chúng. Kết quả sẽ theo định dạng được hướng dẫn, không cần giải thích gì thêm.

---

Cách thực hiện  

1. Xác định thực thể  
   - Tên thực thể (viết hoa chữ cái đầu)  
   - Loại thực thể: Một trong các loại sau:  
     [nhân vật, sự kiện, địa điểm, tổ chức, triều đại, công nghệ, vũ khí, công cụ, xã hội, giai cấp, phong tục, tôn giáo, nghệ thuật, văn hóa, kinh tế, chính trị, quân sự, chủng tộc, hệ tư tưởng]  
   - Mô tả thực thể: Tóm tắt vai trò và đặc điểm  

   Định dạng:  
   `("đối tượng"{tuple_delimiter}<tên thực thể>{tuple_delimiter}<loại thực thể>{tuple_delimiter}<mô tả thực thể>)`

2. Xác định mối quan hệ  
   - Thực thể nguồn → Thực thể đích  
   - Mô tả quan hệ  
   - Mức độ quan hệ (iên điểm)  

   Định dạng:  
   `("đối tượng"{tuple_delimiter}<thực thể nguồn>{tuple_delimiter}<thực thể đích>{tuple_delimiter}<mô tả quan hệ>{tuple_delimiter}<mức độ quan hệ>)`

3. Trả về danh sách thực thể và quan hệ, dùng {record_delimiter} làm dấu phân cách.  

4. Kết thúc bằng {completion_delimiter}.  

---

### Ví dụ đầu ra với dữ liệu lịch sử  

#### Dữ liệu đầu vào  
_Thời kỳ đồ đá mới đánh dấu sự phát triển của công cụ lao động. Con người biết chế tạo rìu mài, cung tên, giúp việc săn bắn hiệu quả hơn. Họ bắt đầu định cư, hình thành các xã hội sơ khai và tạo ra các phong tục mới._  

#### Kết quả đầu ra  
("entity"{tuple_delimiter}"Thời kỳ đồ đá mới"{tuple_delimiter}"sự kiện"{tuple_delimiter}"Giai đoạn tiến bộ của loài người với sự phát triển công cụ lao động và định cư."){record_delimiter}
("entity"{tuple_delimiter}"Rìu mài"{tuple_delimiter}"công cụ"{tuple_delimiter}"Loại công cụ đá được mài nhẵn, giúp con người lao động hiệu quả hơn."){record_delimiter}
("entity"{tuple_delimiter}"Cung tên"{tuple_delimiter}"vũ khí"{tuple_delimiter}"Loại vũ khí giúp săn bắn hiệu quả và an toàn hơn."){record_delimiter}
("entity"{tuple_delimiter}"Xã hội sơ khai"{tuple_delimiter}"xã hội"{tuple_delimiter}"Các cộng đồng người bắt đầu định cư và tổ chức cuộc sống."){record_delimiter}
("relationship"{tuple_delimiter}"Thời kỳ đồ đá mới"{tuple_delimiter}"Rìu mài"{tuple_delimiter}"Thời kỳ đồ đá mới đánh dấu sự xuất hiện của rìu mài."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Cung tên"{tuple_delimiter}"Săn bắn"{tuple_delimiter}"Cung tên giúp việc săn bắn hiệu quả hơn."{tuple_delimiter}10){completion_delimiter}
Văn bản cần xử lý:
{input_text}
---


"""

PROMPTS["validation"] = """
Bạn là một trợ lý AI chuyên gia về trả lời câu hỏi trắc nghiệm. Hãy đọc kỹ câu hỏi sau và chỉ trả lời bằng một ký tự đại diện cho đáp án đúng (A, B, C, D). Không giải thích, không đưa thêm thông tin, chỉ trả về một ký tự duy nhất:
Câu hỏi:
    {question}
Định dạng đầu ra:
- Chỉ trả về một ký tự (A, B, C hoặc D)
"""

PROMPTS["proposition_extract"]= """
Phân tích "Nội dung" thành các mệnh đề rõ ràng và đơn giản, đảm bảo chúng có thể diễn giải được ngoài ngữ cảnh.
1. Chia câu ghép thành các câu đơn giản. Giữ nguyên cách diễn đạt ban đầu từ đầu vào bất cứ khi nào có thể.
2. Đối với bất kỳ thực thể được đặt tên nào đi kèm với thông tin mô tả bổ sung, hãy tách thông tin này thành mệnh đề riêng biệt của nó.
3. Tách mệnh đề khỏi ngữ cảnh bằng cách thêm từ bổ nghĩa cần thiết vào danh từ hoặc toàn bộ câu và thay thế đại từ (ví dụ: "nó", "anh ấy", "cô ấy", "họ", "cái này", "cái kia") bằng tên đầy đủ của các thực thể mà chúng đề cập đến.
4. Trình bày kết quả dưới dạng danh sách các chuỗi, được định dạng trong JSON.
Ví dụ:
*** Đầu vào:
I. Sử học – môn khoa học mang tính liên ngành
- Sử học là ngành khoa học có đối tượng nghiên cứu rộng, liên quan đến nhiều ngành khoa học thuộc nhiều lĩnh vực khác nhau, nhằm phản ánh đầy đủ bức tranh toàn diện của đời sống con người và xã hội loài người.
- Sử học sử dụng tri thức từ các ngành khác nhau để tìm hiểu, nghiên cứu vấn đề một cách toàn diện, hiệu quả, khoa học về con người và xã hội loài người.
- Sử học có khả năng liên kết các môn học, các ngành khoa học với nhau, cả khoa học xã hội nhân văn lẫn khoa học tự nhiên và công nghệ.
II. MỐI LIÊN HỆ GIỮA SỬ HỌC VỚI CÁC NGÀNH KHOA HỌC XÃ HỘI VÀ NHÂN VĂN KHÁC
- Sử học và các ngành khoa học xã hội và nhân văn khác đều lấy xã hội loài người làm đối tượng nghiên cứu, nhưng mỗi khoa học chỉ nghiên cứu một lĩnh vực nhất định. Trong quá trình hình thành và phát triển, Sử học với các ngành khoa học xã hội và nhân văn khác luôn thể hiện mối liên hệ mật thiết với nhau.
- Sử học cung cấp thông tin cho các ngành khoa học xã hội và nhân văn khác về bối cảnh hình thành, phát triển; xác định rõ những nhân tố (chủ quan và khách quan) tác động đến quá trình hình thành, phát triển; dự báo xu hướng vận động phát triển cho các ngành khoa học này.
- Các ngành khoa học xã hội và nhân văn khác có thể hỗ trợ khoa học lịch sử trên các phương diện tri thức, kết quả nghiên cứu,... thành tựu của mỗi ngành khoa học xã hội và nhân văn khác tạo điều kiện, phương tiện, phương pháp,... giúp cho khoa học lịch sử đạt kết quả tốt hơn.

*** Kết quả mong muốn:
[
  "Sử học là ngành khoa học có đối tượng nghiên cứu rộng.",
  "Sử học liên quan đến nhiều ngành khoa học thuộc nhiều lĩnh vực khác nhau.",
  "Mục đích của sử học là phản ánh đầy đủ bức tranh toàn diện của đời sống con người và xã hội loài người.",
  "Sử học sử dụng tri thức từ các ngành khác nhau để nghiên cứu vấn đề về con người và xã hội loài người.",
  "Sử học có khả năng liên kết các môn học và các ngành khoa học với nhau.",
  "Sử học liên kết khoa học xã hội nhân văn với khoa học tự nhiên và công nghệ.",
  "Sử học và các ngành khoa học xã hội và nhân văn đều lấy xã hội loài người làm đối tượng nghiên cứu.",
  "Mỗi ngành khoa học xã hội và nhân văn nghiên cứu một lĩnh vực nhất định.",
  "Sử học với các ngành khoa học xã hội và nhân văn luôn thể hiện mối liên hệ mật thiết với nhau.",
  "Sử học cung cấp thông tin cho các ngành khoa học xã hội và nhân văn khác về bối cảnh hình thành và phát triển.",
  "Sử học xác định rõ những nhân tố tác động đến quá trình hình thành và phát triển.",
  "Sử học dự báo xu hướng vận động phát triển cho các ngành khoa học xã hội và nhân văn.",
  "Các ngành khoa học xã hội và nhân văn có thể hỗ trợ khoa học lịch sử trên các phương diện tri thức và kết quả nghiên cứu.",
  "Thành tựu của các ngành khoa học xã hội và nhân văn giúp khoa học lịch sử đạt kết quả tốt hơn.",
]
*** Đoạn văn cần xử lý:
{paragraph}
"""

PROMPTS["tuple_delimiter"]= "{tuple_delimiter}"
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

