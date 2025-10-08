import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import io
import json
from google import genai
from google.genai.errors import APIError
from google.genai import types 

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Phương án Kinh doanh (Dự án)",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📊")
st.markdown("Tải file Word mô tả dự án và sử dụng AI để trích xuất dữ liệu, tính toán NPV/IRR/PP, và nhận phân tích.")

# Khởi tạo trạng thái phiên cho việc lưu trữ dữ liệu
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None

# --------------------------------------------------------------------------------------
# --- CÁC HÀM XỬ LÝ DỮ LIỆU VÀ API ---
# --------------------------------------------------------------------------------------

def extract_text_from_docx(uploaded_file):
    """Đọc toàn bộ văn bản từ file Word đã tải lên."""
    try:
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Lỗi đọc file Word: {e}")
        return None

# Cấu trúc JSON bắt buộc cho Gemini để đảm bảo đầu ra nhất quán
EXTRACTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "vốn_đầu_tư_ban_đầu": types.Schema(type=types.Type.NUMBER, description="Vốn đầu tư ban đầu (Initial Investment), là số âm."),
        "dòng_đời_dự_án_năm": types.Schema(type=types.Type.INTEGER, description="Dòng đời dự án theo năm (Project Lifespan in Years)."),
        "doanh_thu_hàng_năm": types.Schema(type=types.Type.NUMBER, description="Doanh thu dự kiến hàng năm (Annual Revenue)."),
        "chi_phí_vận_hành_hàng_năm": types.Schema(type=types.Type.NUMBER, description="Tổng chi phí vận hành hàng năm, không bao gồm Khấu hao và Thuế."),
        "wacc_phần_trăm": types.Schema(type=types.Type.NUMBER, description="Chi phí vốn bình quân gia quyền (WACC) dưới dạng phần trăm (ví dụ: 10.5 cho 10.5%)."),
        "thuế_suất_phần_trăm": types.Schema(type=types.Type.NUMBER, description="Thuế suất doanh nghiệp dưới dạng phần trăm (ví dụ: 20 cho 20%).")
    },
    required=[
        "vốn_đầu_tư_ban_đầu", "dòng_đời_dự_án_năm", "doanh_thu_hàng_năm", 
        "chi_phí_vận_hành_hàng_năm", "wacc_phần_trăm", "thuế_suất_phần_trăm"
    ]
)

@st.cache_data(show_spinner="AI đang đọc và trích xuất dữ liệu tài chính...")
def extract_data_with_gemini(doc_text, api_key):
    """
    Sử dụng Gemini API để trích xuất các thông tin tài chính cần thiết 
    từ văn bản và trả về dưới dạng JSON.
    """
    try:
        client = genai.Client(api_key=api_key)
        
        system_prompt = (
            "Bạn là chuyên gia trích xuất dữ liệu tài chính. Nhiệm vụ của bạn là đọc "
            "văn bản mô tả dự án và tìm các chỉ tiêu sau: Vốn đầu tư ban đầu (luôn là số âm), "
            "Dòng đời dự án (số năm), Doanh thu hàng năm, Chi phí vận hành hàng năm (trước thuế), "
            "WACC (phần trăm), và Thuế suất (phần trăm). "
            "Bạn PHẢI trả lời DUY NHẤT bằng một đối tượng JSON tuân thủ schema đã cung cấp. "
            "Nếu không tìm thấy giá trị, hãy cố gắng ước tính hợp lý và ghi chú. "
            "Tất cả giá trị tiền tệ phải được đưa về đơn vị cơ bản (ví dụ: nếu đơn vị là Tỷ, hãy chuyển sang đơn vị thường)."
        )

        prompt = f"Trích xuất các chỉ tiêu tài chính từ văn bản dự án sau:\n\n{doc_text}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=EXTRACTION_SCHEMA,
            )
        )
        
        # Đảm bảo đầu ra là chuỗi JSON hợp lệ và parse nó
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(json_string)
        
        # Chuyển WACC và Thuế suất từ phần trăm sang thập phân
        extracted_data['wacc'] = extracted_data['wacc_phần_trăm'] / 100
        extracted_data['thuế_suất'] = extracted_data['thuế_suất_phần_trăm'] / 100
        
        return extracted_data

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi phân tích JSON: AI trả về định dạng không hợp lệ.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

@st.cache_data(show_spinner="Đang tính toán các chỉ số hiệu quả dự án...")
def calculate_project_metrics(data):
    """
    Xây dựng bảng dòng tiền và tính toán NPV, IRR, PP, DPP.
    Giả định: Không có khấu hao. Tiền thu hồi vốn bằng 0.
    """
    try:
        # Lấy dữ liệu
        I0 = data['vốn_đầu_tư_ban_đầu']
        N = data['dòng_đời_dự_án_năm']
        R = data['doanh_thu_hàng_năm']
        C = data['chi_phí_vận_hành_hàng_năm']
        WACC = data['wacc']
        Tax = data['thuế_suất']
        
        # 1. Xây dựng Dòng tiền Tự do (FCF - Free Cash Flow)
        EBIT = R - C  # Lợi nhuận trước thuế và lãi vay (Giả định không có lãi vay và khấu hao)
        EAT = EBIT * (1 - Tax) # Lợi nhuận sau thuế (Earnings After Tax)
        FCF_annual = EAT # Giả định FCF = EAT (do không có khấu hao và thay đổi vốn lưu động)
        
        # Tạo mảng dòng tiền: Năm 0 là vốn đầu tư, các năm còn lại là FCF
        cash_flows = [I0] + [FCF_annual] * N
        
        # Tạo bảng dòng tiền chi tiết
        years = list(range(N + 1))
        df_cash_flow = pd.DataFrame({
            'Năm': years,
            'Dòng tiền trước thuế (R - C)': [I0] + [R - C] * N,
            'Lợi nhuận sau thuế (EAT)': [I0] + [EAT] * N,
            'Dòng tiền tự do (FCF)': cash_flows,
            'Hệ số chiết khấu': [1 / (1 + WACC)**t for t in years],
            'Dòng tiền chiết khấu': [cf * (1 / (1 + WACC)**t) for t, cf in enumerate(cash_flows)]
        }).set_index('Năm')

        # 2. Tính toán các chỉ số
        
        # NPV (Net Present Value)
        # np.npv(rate, values)
        npv = np.npv(WACC, cash_flows)
        
        # IRR (Internal Rate of Return)
        # np.irr(values)
        try:
            irr = np.irr(cash_flows)
        except ValueError:
             irr = np.nan # NaN nếu không có IRR thực

        # PP (Payback Period - Thời gian hoàn vốn)
        cumulative_cf = np.cumsum(cash_flows)
        pp = np.nan
        for i in range(1, len(cumulative_cf)):
            if cumulative_cf[i] >= 0 and cumulative_cf[i-1] < 0:
                pp = i - 1 + abs(cumulative_cf[i-1]) / cash_flows[i]
                break
        
        # DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
        discounted_cf = df_cash_flow['Dòng tiền chiết khấu'].values
        cumulative_dcf = np.cumsum(discounted_cf)
        dpp = np.nan
        for i in range(1, len(cumulative_dcf)):
            if cumulative_dcf[i] >= 0 and cumulative_dcf[i-1] < 0:
                dpp = i - 1 + abs(cumulative_dcf[i-1]) / discounted_cf[i]
                break
                
        metrics = {
            'NPV': npv,
            'IRR': irr,
            'PP': pp,
            'DPP': dpp,
            'df_cash_flow': df_cash_flow
        }
        return metrics

    except Exception as e:
        st.error(f"Lỗi tính toán tài chính: {e}")
        return None

def get_ai_financial_analysis(metrics, data, api_key):
    """Gửi các chỉ số và dữ liệu nền tảng đến Gemini để phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        
        metrics_df = pd.DataFrame([metrics]).drop(columns=['df_cash_flow'])
        
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính đầu tư có kinh nghiệm.
        
        Dữ liệu đầu vào: {data}
        
        Kết quả tính toán hiệu quả dự án:
        {metrics_df.to_markdown(index=False)}
        
        WACC (Chi phí vốn) được sử dụng là: {data['wacc'] * 100:.2f}%
        
        Yêu cầu:
        1. Nhận xét về tính khả thi của dự án DỰA TRÊN NPV (phải là số dương), IRR (phải lớn hơn WACC) và thời gian hoàn vốn (PP, DPP).
        2. Đánh giá xem dự án có nên được chấp nhận hay không.
        3. Đưa ra 1-2 khuyến nghị ngắn gọn nếu dự án có điểm yếu (ví dụ: hoàn vốn chậm, IRR thấp).
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --------------------------------------------------------------------------------------
# --- BẮT ĐẦU GIAO DIỆN STREAMLIT ---
# --------------------------------------------------------------------------------------

# --- Chức năng Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa Phương án Kinh doanh",
    type=['docx']
)

# Lấy Khóa API 
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

if uploaded_file is not None and api_key:
    
    # --- Chức năng 1: Lọc Dữ liệu bằng AI ---
    st.subheader("2. Trích xuất Dữ liệu và Tính toán")
    
    if st.button("▶️ 1. Lọc dữ liệu từ File Word (Sử dụng AI)", key="extract_button"):
        with st.spinner('Đang đọc file và AI đang trích xuất các chỉ tiêu...'):
            doc_text = extract_text_from_docx(uploaded_file)
            if doc_text:
                extracted_data = extract_data_with_gemini(doc_text, api_key)
                
                if extracted_data:
                    st.session_state["extracted_data"] = extracted_data
                    
                    # Tự động tính toán sau khi trích xuất thành công
                    st.session_state["metrics"] = calculate_project_metrics(extracted_data)
                    
                    st.success("✅ Trích xuất và Tính toán thành công!")
                
if st.session_state["extracted_data"]:
    
    # Hiển thị dữ liệu đầu vào đã được chuẩn hóa
    st.markdown("---")
    st.subheader("2.1. Dữ liệu Đầu vào Đã Lọc")
    
    display_data = {
        "Vốn đầu tư ban đầu": f"{st.session_state['extracted_data']['vốn_đầu_tư_ban_đầu']:,.0f}",
        "Dòng đời dự án": f"{st.session_state['extracted_data']['dòng_đời_dự_án_năm']} năm",
        "Doanh thu hàng năm": f"{st.session_state['extracted_data']['doanh_thu_hàng_năm']:,.0f}",
        "Chi phí vận hành hàng năm": f"{st.session_state['extracted_data']['chi_phí_vận_hành_hàng_năm']:,.0f}",
        "WACC (Chi phí vốn)": f"{st.session_state['extracted_data']['wacc'] * 100:.2f}%",
        "Thuế suất": f"{st.session_state['extracted_data']['thuế_suất'] * 100:.2f}%",
    }
    
    st.dataframe(pd.DataFrame(list(display_data.items()), columns=['Chỉ tiêu', 'Giá trị']), hide_index=True)
    
    # --- Chức năng 2 & 3: Bảng Dòng tiền và Chỉ số Đánh giá ---
    if st.session_state["metrics"]:
        
        metrics = st.session_state["metrics"]
        df_cf = metrics['df_cash_flow']
        
        st.markdown("---")
        st.subheader("2.2. Bảng Dòng tiền Dự án (Cash Flow)")
        st.dataframe(df_cf.style.format({
            'Dòng tiền trước thuế (R - C)': '{:,.0f}',
            'Lợi nhuận sau thuế (EAT)': '{:,.0f}',
            'Dòng tiền tự do (FCF)': '{:,.0f}',
            'Hệ số chiết khấu': '{:.4f}',
            'Dòng tiền chiết khấu': '{:,.0f}',
        }), use_container_width=True)
        
        st.markdown("---")
        st.subheader("3. Các Chỉ số Đánh giá Hiệu quả Dự án")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="NPV (Giá trị hiện tại ròng)",
                value=f"{metrics['NPV']:,.0f}" if not np.isnan(metrics['NPV']) else "Lỗi",
                delta="Dương: Chấp nhận" if metrics['NPV'] > 0 else "Âm: Từ chối" if metrics['NPV'] < 0 else "Bằng 0"
            )

        with col2:
            st.metric(
                label="IRR (Tỷ suất sinh lời nội bộ)",
                value=f"{metrics['IRR'] * 100:.2f}%" if not np.isnan(metrics['IRR']) else "Không tính được",
                delta="> WACC: Chấp nhận" if metrics['IRR'] > st.session_state['extracted_data']['wacc'] else "< WACC: Từ chối" if not np.isnan(metrics['IRR']) else None
            )

        with col3:
            st.metric(
                label="PP (Thời gian hoàn vốn)",
                value=f"{metrics['PP']:.2f} năm" if not np.isnan(metrics['PP']) else "Không hoàn vốn"
            )

        with col4:
            st.metric(
                label="DPP (Thời gian hoàn vốn có chiết khấu)",
                value=f"{metrics['DPP']:.2f} năm" if not np.isnan(metrics['DPP']) else "Không hoàn vốn"
            )

        # --- Chức năng 4: Phân tích AI ---
        st.markdown("---")
        st.subheader("4. Yêu cầu AI Phân tích Hiệu quả Dự án")
        
        if st.button("🤖 Phân tích các Chỉ số (NPV, IRR, PP, DPP)", key="analysis_button"):
            with st.spinner('Đang gửi các chỉ số tới Gemini để nhận xét chuyên sâu...'):
                ai_result = get_ai_financial_analysis(
                    metrics, 
                    st.session_state["extracted_data"], 
                    api_key
                )
                st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                st.info(ai_result)

else:
    # Hướng dẫn khi chưa có file
    if not api_key:
        st.warning("Vui lòng thiết lập API Key để sử dụng các chức năng AI.")
    st.info("Để bắt đầu, vui lòng tải lên một file Word (.docx) chứa các thông tin về vốn đầu tư, doanh thu, chi phí, WACC và dòng đời dự án.")
