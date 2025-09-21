import cv2
import pytesseract
import re
from PIL import Image

# Tesseract installation path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class MedicalOCRPipeline:
    def __init__(self, lang="eng"):
        self.lang = lang

        # Pre-compiled regex patterns for performance
        self.patterns = {
            "Age": r"Age\s*[:\-]?\s*(\d+)",
            "Sex": r"Sex\s*[:\-]?\s*(\w+)",
            "Blood Pressure": r"Blood[_\s]?Pressure\s*[:\-]?\s*(\d+/?\d*)",
            "Cholesterol": r"Cholesterol\s*[:\-]?\s*(\d+)",
            "Blood Sugar": r"Blood[_\s]?Sugar\s*[:\-]?\s*(\d+)",
            "ECG": r"Ecg\s*[:\-]?\s*(\w+)",
            "Max Heart Rate": r"Maximum Heart Rate\s*[:\-]?\s*(\d+)",
            "Exercise Angina": r"Exercise[_\s]?Angina\s*[:\-]?\s*(\w+)",
            "ST Depression": r"ST[_\s]?Depression\s*[:\-]?\s*([\d.]+)",
            "ST Slope": r"ST[_\s]?Slope\s*[:\-]?\s*(\w+)"
        }

    def preprocess_image(self, image_path):
        """Load and preprocess the image for OCR."""
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding for noise removal
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def extract_text(self, image_path):
        """Perform OCR on the image and return extracted text."""
        processed_img = self.preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_img, lang=self.lang, config="--psm 6")
        return text

    def parse_medical_values(self, text):
        """Parse structured medical values from OCR text."""
        results = {}
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[field] = match.group(1)
        return results

    def run_pipeline(self, image_path):
        """Run the full OCR pipeline: preprocess → OCR → parse → structured dict."""
        text = self.extract_text(image_path)
        parsed_data = self.parse_medical_values(text)
        return {"raw_text": text, "parsed_data": parsed_data}