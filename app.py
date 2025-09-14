import os
import json
import tempfile
import io
import re
import warnings
import pandas as pd
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Настройки
st.set_page_config(
    page_title="Document Intelligence System", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📄 Advanced Document Processing System")
st.write("Загрузите документ для извлечения структурированных данных")

load_dotenv()

# Конфигурация Gemini AI
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Кэшированные модели
@st.cache_resource
def load_easyocr():
    """Загрузка EasyOCR с кэшированием"""
    try:
        import easyocr
        st.sidebar.info("🔄 Загрузка OCR модели...")
        reader = easyocr.Reader(['en', 'ru'], gpu=False)
        st.sidebar.success("✅ OCR модель загружена!")
        return reader
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка загрузки OCR: {e}")
        return None

@st.cache_resource
def load_gemini_model(api_key):
    """Загрузка Gemini модели с кэшированием"""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Получаем список доступных моделей
        available_models = genai.list_models()
        model_names = [model.name for model in available_models]
        
        # Выбираем подходящую модель
        gemini_model_name = None
        preferred_models = [
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        for model in preferred_models:
            if model in model_names:
                gemini_model_name = model
                break
        
        if gemini_model_name:
            st.sidebar.success(f"✅ Используется модель: {gemini_model_name}")
            return genai.GenerativeModel(gemini_model_name)
        else:
            st.sidebar.warning("⚠️ Доступные модели Gemini: " + ", ".join(model_names))
            return None
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Gemini не доступен: {e}")
        return None

class DocumentProcessor:
    def __init__(self, ocr_reader, gemini_model):
        self.ocr_reader = ocr_reader
        self.gemini_model = gemini_model
    
    def preprocess_image(self, image):
        """Предобработка изображения"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Улучшение контраста и резкости
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.8)
        
        # Оптимальный размер для распознавания
        if min(image.size) < 1000:
            scale = 1500 / min(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image
    
    def extract_text_easyocr(self, image):
        """Извлечение текста с помощью EasyOCR"""
        try:
            image_np = np.array(image)
            
            # Оптимальный размер для OCR
            if max(image_np.shape) > 2000:
                scale = 2000 / max(image_np.shape)
                new_size = (int(image_np.shape[1] * scale), int(image_np.shape[0] * scale))
                image_np = np.array(image.resize(new_size, Image.LANCZOS))
            
            results = self.ocr_reader.readtext(
                image_np, 
                detail=0,
                paragraph=True,
                contrast_ths=0.2,
                adjust_contrast=0.6
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
            {text[:2000]}
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
            Ты - эксперт по анализу документов. Проанализируй текст документа и извлеки структурированные данные.
            Верни результат в формате JSON строго со следующими полями:
            
            {{
                "document_type": "тип документа (invoice, contract, receipt, unknown)",
                "sender": "отправитель/продавец",
                "receiver": "получатель/покупатель", 
                "date": "дата документа",
                "amount": "сумма/стоимость",
                "description": "описание/назначение платежа",
                "contract_number": "номер договора если есть",
                "currency": "валюта если определена"
            }}
            
            Если какое-то поле невозможно определить, используй "unknown".
            Важно: верни только чистый JSON без каких-либо дополнительных комментариев.
            
            Текст документа для анализа:
            {text[:2500]}
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Извлечение JSON из ответа
            response_text = response.text.strip()
            
            # Убираем markdown коды если есть
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as je:
                    st.warning(f"Не удалось распарсить JSON от Gemini: {je}")
                    # Пробуем почистить JSON
                    try:
                        # Убираем лишние символы
                        json_str_clean = re.sub(r'[^\x00-\x7F]+', '', json_str)
                        return json.loads(json_str_clean)
                    except:
                        pass
            
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
                "currency": self._extract_currency(text),
                "raw_text_preview": text[:400] + "..." if len(text) > 400 else text
            }
            
            return structured_data
            
        except Exception as e:
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
        elif any(word in text_lower for word in ['receipt', 'чек', 'квитанция']):
            return "receipt"
        else:
            return "unknown"
    
    def _extract_date(self, text):
        """Извлечение даты"""
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}-\d{2}-\d{2}'
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
            r'сумма[:\s]*(\d+[,\d]*\.?\d*)',
            r'amount[:\s]*(\d+[,\d]*\.?\d*)',
            r'total[:\s]*(\d+[,\d]*\.?\d*)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else str(matches[0])
        return "unknown"
    
    def _extract_currency(self, text):
        """Извлечение валюты"""
        currency_patterns = [
            r'\$', r'USD', r'EUR', r'RUB', r'₽', r'тенге', r'KZT'
        ]
        
        for pattern in currency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern.replace('\\', '')
        return "unknown"
    
    def _extract_description(self, text):
        """Извлечение описания"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['description', 'описание', 'назначение']):
                return line
        return lines[2] if len(lines) > 2 else "unknown"
    
    def _extract_contract_number(self, text):
        """Извлечение номера договора"""
        patterns = [
            r'№\s*[A-Za-z0-9-]+',
            r'No\.?\s*[A-Za-z0-9-]+',
            r'contract no\.?\s*[A-Za-z0-9-]+',
            r'договор №\s*[A-Za-z0-9-]+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return "unknown"

def process_pdf_file(uploaded_file):
    """Обработка PDF файла"""
    try:
        pdf_data = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        page = doc.load_page(0)
        
        zoom = 1.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data))
        doc.close()
        
        return image
        
    except Exception as e:
        raise e

def process_excel_file(uploaded_file):
    """Обработка Excel файла"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl', nrows=100)  # Ограничиваем строки
        elif uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, nrows=100)
        else:  # CSV
            df = pd.read_csv(uploaded_file, nrows=100)
        
        text = "Excel Document Data:\n\n"
        for col in df.columns:
            sample_values = df[col].dropna().head(5).astype(str).tolist()
            text += f"{col}: {', '.join(sample_values)}\n"
        
        return text, df
        
    except Exception as e:
        raise e

def process_text_file(uploaded_file):
    """Обработка текстовых файлов"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        return content[:5000]  # Ограничиваем размер
    except:
        try:
            content = uploaded_file.getvalue().decode('latin-1')
            return content[:5000]
        except:
            return "Не удалось декодировать текстовый файл"

def process_word_file(uploaded_file):
    """Обработка Word документов"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.getvalue()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs[:50]])  # Ограничиваем
        return text
    except Exception as e:
        return f"Ошибка обработки Word документа: {str(e)}"

def main():
    # Загрузка моделей с кэшированием
    gemini_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=GEMINI_API_KEY if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY" else "",
        type="password", 
        help="Получите API ключ на https://aistudio.google.com/"
    )
    
    # Загрузка моделей
    ocr_reader = load_easyocr()
    gemini_model = load_gemini_model(gemini_key)
    
    if not ocr_reader:
        st.error("Не удалось загрузить OCR модель. Пожалуйста, проверьте установку EasyOCR.")
        return
    
    # Инициализация процессора
    if 'processor' not in st.session_state or st.session_state.get('gemini_key') != gemini_key:
        processor = DocumentProcessor(ocr_reader, gemini_model)
        st.session_state.processor = processor
        st.session_state.gemini_key = gemini_key
    
    processor = st.session_state.processor
    
    # Загрузка файла
    st.sidebar.header("📁 Загрузка документа")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите файл",
        type=[
            'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp', 'gif',  # Изображения
            'xlsx', 'xls', 'csv',  # Excel
            'txt', 'doc', 'docx', 'rtf',  # Текстовые документы
        ],
        help="Поддерживаются все основные форматы файлов"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Загруженный документ")
            st.info(f"📝 Тип файла: {uploaded_file.type}")
            st.info(f"📊 Размер: {uploaded_file.size} байт")
            
            try:
                file_content = None
                additional_data = None
                
                # Обработка разных типов файлов
                if uploaded_file.type.startswith('image/'):
                    # Изображения
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Загруженное изображение", width=400)
                    file_content = image
                    
                elif uploaded_file.type == 'application/pdf':
                    # PDF файлы
                    image = process_pdf_file(uploaded_file)
                    st.image(image, caption="Первая страница PDF", width=400)
                    file_content = image
                    
                elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                                          'application/vnd.ms-excel', 'text/csv']:
                    # Excel/CSV файлы
                    text, df = process_excel_file(uploaded_file)
                    st.dataframe(df.head(), use_container_width=True)
                    file_content = text
                    additional_data = df
                    
                elif uploaded_file.type in ['text/plain', 'application/rtf']:
                    # Текстовые файлы
                    text = process_text_file(uploaded_file)
                    st.text_area("Содержимое файла:", text, height=200, label_visibility="visible")
                    file_content = text
                    
                elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                          'application/msword']:
                    # Word документы
                    text = process_word_file(uploaded_file)
                    st.text_area("Текст документа:", text, height=200, label_visibility="visible")
                    file_content = text
                    
                else:
                    st.warning("Формат файла не поддерживается для автоматической обработки")
                    file_content = None
                
                # Обработка документа
                if file_content and st.button("🚀 Обработать документ", type="primary", use_container_width=True):
                    with st.spinner("⏳ Обработка..."):
                        if isinstance(file_content, Image.Image):
                            # OCR обработка изображений
                            extracted_text = processor.extract_text_easyocr(file_content)
                            
                            # Улучшение текста с Gemini
                            if processor.gemini_model:
                                with st.spinner("🤖 Улучшение текста с AI..."):
                                    enhanced_text = processor.enhance_text_with_gemini(extracted_text)
                            else:
                                enhanced_text = extracted_text
                            
                            # Структурирование данных
                            with st.spinner("📊 Извлечение структурированных данных..."):
                                if processor.gemini_model:
                                    structured_data = processor.extract_structured_data_with_gemini(enhanced_text)
                                else:
                                    structured_data = processor.extract_structured_data_basic(enhanced_text)
                            
                            with col2:
                                st.subheader("🔍 Результаты OCR обработки")
                                with st.expander("📝 Распознанный текст"):
                                    st.text_area("Текст:", extracted_text, height=200, label_visibility="visible")
                                
                                if processor.gemini_model:
                                    with st.expander("✨ Текст после улучшения AI"):
                                        st.text_area("Улучшенный текст:", enhanced_text, height=200, label_visibility="visible")
                            
                        else:
                            # Обработка текстовых данных
                            if processor.gemini_model:
                                with st.spinner("🤖 Улучшение текста с AI..."):
                                    enhanced_text = processor.enhance_text_with_gemini(file_content)
                                with st.spinner("📊 Извлечение структурированных данных..."):
                                    structured_data = processor.extract_structured_data_with_gemini(enhanced_text)
                            else:
                                with st.spinner("📊 Извлечение структурированных данных..."):
                                    structured_data = processor.extract_structured_data_basic(file_content)
                            
                            with col2:
                                st.subheader("✅ Результаты обработки")
                                st.success("Текстовый документ успешно обработан")
                        
                        # Показ структурированных данных
                        st.subheader("📊 Структурированные данные")
                        st.json(structured_data)
                        
                        # Скачивание результата
                        json_str = json.dumps(structured_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="💾 Скачать JSON результат",
                            data=json_str,
                            file_name="processed_document.json",
                            mime="application/json",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Ошибка обработки: {str(e)}")
    
    # Инструкция
    st.sidebar.info("""
    **📋 Поддерживаемые форматы:**
    - 📷 Изображения: PNG, JPG, JPEG, TIFF, BMP, GIF
    - 📄 Документы: PDF, TXT, DOC, DOCX, RTF
    - 📊 Таблицы: XLSX, XLS, CSV
    
    **🔑 Для использования Gemini AI:**
    1. Получите API ключ на https://aistudio.google.com/
    2. Введите ключ в поле выше
    3. Наслаждайтесь улучшенным распознаванием!
    """)

if __name__ == "__main__":
    import numpy as np
    main()