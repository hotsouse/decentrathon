import os
import json
import tempfile
import io
import re
import warnings
import requests
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import torch
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import os


# Настройки
st.set_page_config(page_title="Document Intelligence System", layout="wide")
st.title("📄 Advanced Document Processing System")
st.write("Загрузите документ для извлечения структурированных данных")

load_dotenv()

# Конфигурация Gemini AI
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # Замените на ваш API ключ

class DocumentProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Используемое устройство: {self.device}")
        self._initialize_models()
        
    def _initialize_models(self):
        """Инициализация моделей"""
        try:
            # Инициализация Gemini AI
            st.write("Инициализация Gemini AI...")
            if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            else:
                st.warning("Gemini API ключ не настроен. Используется базовое распознавание.")
                self.gemini_model = None
            
            # Инициализация EasyOCR
            st.write("Инициализация OCR...")
            import easyocr
            
            self.easy_reader = easyocr.Reader(
                ['en', 'ru'], 
                gpu=self.device == "cuda"
            )
            
            st.success("Модели успешно загружены!")
            
        except Exception as e:
            st.error(f"Ошибка инициализации моделей: {e}")
            raise e
    
    def preprocess_image(self, image):
        """Предобработка изображения"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Улучшение контраста и резкости
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Увеличение размера для лучшего распознавания
        if min(image.size) < 1200:
            new_size = (image.size[0] * 2, image.size[1] * 2)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Конвертация в grayscale для лучшего OCR
        image_gray = image.convert('L')
        return image_gray
    
    def extract_text_easyocr(self, image):
        """Извлечение текста с помощью EasyOCR"""
        try:
            image_np = np.array(image)
            
            # Оптимальный размер для OCR
            if max(image_np.shape) > 2500:
                scale = 2500 / max(image_np.shape)
                new_size = (int(image_np.shape[1] * scale), int(image_np.shape[0] * scale))
                image_np = np.array(image.resize(new_size, Image.LANCZOS))
            
            # Детальное распознавание с параметрами для лучшего качества
            results = self.easy_reader.readtext(
                image_np, 
                detail=0,
                paragraph=True,
                contrast_ths=0.1,
                adjust_contrast=0.7
            )
            
            return "\n".join(results)
        except Exception as e:
            st.warning(f"OCR error: {e}")
            return ""
    
    def enhance_text_with_gemini(self, text):
        """Улучшение текста с помощью Gemini AI"""
        if not self.gemini_model or not text.strip():
            return text
            
        try:
            prompt = f"""
            Исправь орфографические ошибки и улучши читаемость следующего текста, 
            сохраняя оригинальное содержание и структуру. Верни только исправленный текст.
            
            Текст для исправления:
            {text}
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini enhancement error: {e}")
            return text
    
    def extract_structured_data_with_gemini(self, text):
        """Извлечение структурированных данных с помощью Gemini AI"""
        if not self.gemini_model:
            return self.extract_structured_data_basic(text)
            
        try:
            prompt = f"""
            Проанализируй текст документа и извлеки структурированные данные в формате JSON.
            Верни ТОЛЬКО JSON без каких-либо дополнительных объяснений.
            
            Поля JSON:
            - document_type (тип документа: invoice, contract, receipt, unknown)
            - sender (отправитель/продавец)
            - receiver (получатель/покупатель)
            - date (дата документа)
            - amount (сумма/стоимость)
            - description (описание/назначение платежа)
            - contract_number (номер договора, если есть)
            - items (список товаров/услуг, если есть)
            
            Если какое-то поле невозможно определить, используй "unknown".
            
            Текст документа:
            {text[:3000]}
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Извлечение JSON из ответа
            response_text = response.text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    st.warning("Не удалось распарсить JSON от Gemini")
            
            return self.extract_structured_data_basic(text)
            
        except Exception as e:
            st.warning(f"Gemini structuring error: {e}")
            return self.extract_structured_data_basic(text)
    
    def extract_structured_data_basic(self, text):
        """Базовое извлечение структурированных данных"""
        try:
            structured_data = {
                "document_type": self._detect_document_type(text),
                "sender": self._extract_field(text, ['from:', 'от:', 'sender:', 'отправитель:', 'продавец:']),
                "receiver": self._extract_field(text, ['to:', 'кому:', 'receiver:', 'получатель:', 'покупатель:']),
                "date": self._extract_date(text),
                "amount": self._extract_amount(text),
                "description": self._extract_description(text),
                "contract_number": self._extract_contract_number(text),
                "raw_text_preview": text[:500] + "..." if len(text) > 500 else text
            }
            
            return structured_data
            
        except Exception as e:
            st.warning(f"Structuring error: {e}")
            return {"error": str(e), "raw_text": text}
    
    def _extract_field(self, text, keywords):
        """Извлечение поля по ключевым словам"""
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if keyword in line_lower:
                    return line.replace(keyword, '').replace(keyword.upper(), '').strip()
        return "unknown"
    
    def _detect_document_type(self, text):
        """Определение типа документа"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['invoice', 'счет', 'счёт', 'инвойс']):
            return "invoice"
        elif any(word in text_lower for word in ['contract', 'договор', 'контракт']):
            return "contract"
        elif any(word in text_lower for word in ['receipt', 'чек', 'квитанция', 'receipt']):
            return "receipt"
        else:
            return "unknown"
    
    def _extract_date(self, text):
        """Извлечение даты"""
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+[а-я]+\s+\d{4}',
            r'\d{1,2}\s+[a-z]+\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return "unknown"
    
    def _extract_amount(self, text):
        """Извлечение суммы"""
        amount_patterns = [
            r'\$\s*\d+[,\d]*\.?\d*',
            r'\d+[,\d]*\.?\d*\s*USD',
            r'\d+[,\d]*\.?\d*\s*EUR',
            r'\d+[,\d]*\.?\d*\s*RUB',
            r'\d+[,\d]*\.?\d*\s*₽',
            r'\d+[,\d]*\.?\d*\s*тенге',
            r'сумма[:\s]*(\d+[,\d]*\.?\d*)',
            r'amount[:\s]*(\d+[,\d]*\.?\d*)',
            r'total[:\s]*(\d+[,\d]*\.?\d*)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else str(matches[0])
        return "unknown"
    
    def _extract_description(self, text):
        """Извлечение описания"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['description', 'описание', 'назначение', 'предмет']):
                return line
        return lines[2] if len(lines) > 2 else "unknown"
    
    def _extract_contract_number(self, text):
        """Извлечение номера договора"""
        patterns = [
            r'№\s*[A-Za-z0-9-]+',
            r'No\.?\s*[A-Za-z0-9-]+',
            r'contract no\.?\s*[A-Za-z0-9-]+',
            r'договор №\s*[A-Za-z0-9-]+',
            r'number\s*[A-Za-z0-9-]+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return "unknown"
    
    def process_document(self, image):
        """Основной метод обработки документа"""
        # Предобработка
        processed_image = self.preprocess_image(image)
        
        # Извлечение текста
        with st.spinner("Распознавание текста..."):
            extracted_text = self.extract_text_easyocr(processed_image)
        
        # Улучшение текста с Gemini
        if self.gemini_model:
            with st.spinner("Улучшение текста с AI..."):
                enhanced_text = self.enhance_text_with_gemini(extracted_text)
        else:
            enhanced_text = extracted_text
        
        # Структурирование данных
        with st.spinner("Извлечение структурированных данных..."):
            if self.gemini_model:
                structured_data = self.extract_structured_data_with_gemini(enhanced_text)
            else:
                structured_data = self.extract_structured_data_basic(enhanced_text)
        
        return {
            "raw_text": extracted_text,
            "enhanced_text": enhanced_text,
            "structured_data": structured_data
        }

def process_pdf_file(uploaded_file):
    """Обработка PDF файла"""
    try:
        # Читаем PDF из памяти
        pdf_data = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        page = doc.load_page(0)
        
        # Увеличиваем разрешение для лучшего качества
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data))
        doc.close()
        
        return image
        
    except Exception as e:
        raise e

def main():
    # Настройка Gemini API ключа
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                      help="Получите API ключ на https://aistudio.google.com/")
    
    # Инициализация процессора
    if 'processor' not in st.session_state or st.session_state.get('gemini_key') != gemini_key:
        with st.spinner("Загрузка моделей..."):
            try:
                global GEMINI_API_KEY
                if gemini_key:
                    GEMINI_API_KEY = gemini_key
                
                st.session_state.processor = DocumentProcessor()
                st.session_state.gemini_key = gemini_key
            except Exception as e:
                st.error(f"Не удалось инициализировать процессор: {e}")
                return
    
    processor = st.session_state.processor
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите документ (изображение или PDF)",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Поддерживаются сканы, фотографии, PDF-файлы"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Загруженный документ")
            
            try:
                if uploaded_file.type == "application/pdf":
                    image = process_pdf_file(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                
                st.image(image, caption="Загруженный документ", width='stretch')
                
                if st.button("Обработать документ", type="primary"):
                    result = processor.process_document(image)
                    
                    with col2:
                        st.subheader("Результаты обработки")
                        
                        # Исходный текст
                        with st.expander("Исходный распознанный текст"):
                            st.text_area(
                                "Исходный текст:", 
                                result['raw_text'], 
                                height=200,
                                label_visibility="visible"
                            )
                        
                        # Улучшенный текст
                        if processor.gemini_model:
                            with st.expander("Текст после улучшения AI"):
                                st.text_area(
                                    "Улучшенный текст:", 
                                    result['enhanced_text'], 
                                    height=200,
                                    label_visibility="visible"
                                )
                        
                        # Структурированные данные
                        st.subheader("Структурированные данные")
                        st.json(result['structured_data'])
                        
                        # Скачать результат
                        json_str = json.dumps(result['structured_data'], ensure_ascii=False, indent=2)
                        st.download_button(
                            label="Скачать JSON",
                            data=json_str,
                            file_name="processed_document.json",
                            mime="application/json"
                        )
                        
            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    main()