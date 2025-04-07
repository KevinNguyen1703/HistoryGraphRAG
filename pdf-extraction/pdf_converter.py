import fitz
import cv2
import pytesseract
import os

def pdf2img(pdf_file = "dataset/history_10.pdf", output_dir= "datast/10-old") -> None:
    doc = fitz.open(pdf_file)
    zoom = 4
    mat = fitz.Matrix(zoom, zoom)
    # Count variable is to get the number of pages in the pdf
    for i in range(len(doc)):
        # val = f"image_{i+1}.png"
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        pix.save(f"{output_dir}/page_{i+1}.png")
    doc.close()

def img2text(img_dir = "dataset/10-old", output_dir = "dataset/10-old") -> None:
    os.makedirs(output_dir, exist_ok=True)
    custom_oem_psm_config = r'--oem 3 --psm 6 -l vie'

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in image_files:
        image_path = os.path.join(img_dir, image_file)
        text_filename = os.path.splitext(image_file)[0] + ".txt"  # Convert image_1.png → text_1.txt
        text_output_path = os.path.join(output_dir, text_filename)

        print(f"Processing: {image_path} -> {text_output_path}")

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding (optional, improves OCR results)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform OCR
        text = pytesseract.image_to_string(gray, config=custom_oem_psm_config)

        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"OCR completed. Extracted {img_dir} saved in {output_dir}")

if __name__ == "__main__":
    PDF_FILE= "../dataset/book/sach-giao-khoa-lich-su-10-chan-troi-sang-tao.pdf"
    IMAGE_DIR= "../dataset/image/10-chan-troi-sang-tao"
    TEXT_DIR= "../dataset/text-extracted/10-chan-troi-sang-tao"
    print("Converting pdf book into image")
    pdf2img(pdf_file=PDF_FILE,output_dir=IMAGE_DIR)
    print("Converting image to text")
    img2text(img_dir=IMAGE_DIR,output_dir=TEXT_DIR)