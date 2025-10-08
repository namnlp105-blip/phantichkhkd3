# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from google.genai import types 

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --------------------------------------------------------------------------------------
# --- CÁC HÀM XỬ LÝ DỮ LIỆU VÀ API ---
# --------------------------------------------------------------------------------------

@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # 1. Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 2. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 3. Tính Tỷ trọng theo Tổng Tài sản/Nguồn vốn
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý chia cho 0 để tính Tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tĩnh."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

def get_chat_session(api_key, system_prompt):
    """Khởi tạo hoặc lấy chat session từ st.session_state."""
    # CHỈ KHỞI TẠO NẾU CHƯA CÓ SESSION
    if "chat_client" not in st.session_state:
        try:
            client = genai.Client(api_key=api_key) 
            
            # Khởi tạo model config với System Instruction (đã sửa lỗi 'unexpected keyword')
            config = types.GenerateContentConfig(
                system_instruction=system_prompt
            )
            
            st.session_state.chat_client = client.chats.create(
                model='gemini-2.5-flash',
                config=config 
            )
            # Khởi tạo tin nhắn welcome (chỉ chạy lần đầu)
            st.session_state.messages.append({"role": "assistant", "content": "Chào bạn! Tôi đã phân tích sơ bộ dữ liệu. Hãy hỏi tôi về tăng trưởng, cơ cấu tài sản, hoặc khả năng thanh toán."})

        except Exception as e:
            st.error(f"Lỗi khởi tạo Chat: Vui lòng kiểm tra API Key. Chi tiết: {e}")
            return None
            
    return st.session_state.chat_client

# --------------------------------------------------------------------------------------
# --- BẮT ĐẦU GIAO DIỆN STREAMLIT ---
# --------------------------------------------------------------------------------------

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo trạng thái phiên cho chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if uploaded_file is not None:
    # Lấy Khóa API 
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    else:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            df_processed = process_financial_data(df_raw.copy())

            if df_processed is not None:
                
                # --- Chức năng 2 & 3: Hiển thị Kết quả Tăng trưởng & Tỷ trọng ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # --- Chức năng 4: Tính Chỉ số Tài chính Cơ bản ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                
                try:
                    # Lấy dữ liệu 
                    tsnh_n_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
                    no_ngan_han_row = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]
                    
                    if tsnh_n_row.empty or no_ngan_han_row.empty:
                        raise IndexError("Thiếu dữ liệu để tính chỉ số Thanh toán Hiện hành.")

                    tsnh_n = tsnh_n_row['Năm sau'].iloc[0]
                    tsnh_n_1 = tsnh_n_row['Năm trước'].iloc[0]
                    no_ngan_han_N = no_ngan_han_row['Năm sau'].iloc[0]  
                    no_ngan_han_N_1 = no_ngan_han_row['Năm trước'].iloc[0]

                    # Tính toán (Đã sửa lỗi đánh máy)
                    if no_ngan_han_N != 0:
                        thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    if no_ngan_han_N_1 != 0:
                        thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 # <-- Đã Sửa Lỗi Đánh Máy
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"
                        )
                    with col2:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A",
                            delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                        )
                        
                except IndexError as ie:
                    st.warning(f"Thiếu chỉ tiêu: {ie}")
                except ZeroDivisionError:
                    st.warning("Mẫu số (Nợ Ngắn Hạn) bằng 0, không thể tính chỉ số Thanh toán Hiện hành.")
                    thanh_toan_hien_hanh_N = "N/A"
                    thanh_toan_hien_hanh_N_1 = "N/A"
                    
                # Chuẩn bị dữ liệu cho AI (đã sửa lỗi cú pháp f-string)
                data_for_ai_markdown = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        df_processed.to_markdown(index=False),
                        (f"{thanh_toan_hien_hanh_N_1}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"), 
                        (f"{thanh_toan_hien_hanh_N}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A")
                    ]
                }).to_markdown(index=False)
                
                # --- Chức năng 5: Nhận xét AI (Statis Analysis) ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI Tĩnh)")
                
                if st.button("Yêu cầu AI Phân tích"):
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai_markdown, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)

                # --------------------------------------------------------------------------------------
                # --- CHỨC NĂNG 6: KHUNG CHAT HỎI ĐÁP VỚI GEMINI (Đã Tích hợp) ---
                # --------------------------------------------------------------------------------------
                st.subheader("6. Chat Hỏi đáp chuyên sâu với Gemini AI 💬")
                
                SYSTEM_PROMPT = f"""
                Bạn là một trợ lý phân tích tài chính chuyên nghiệp và lịch sự.
                Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên dữ liệu Báo cáo Tài chính sau.
                Bạn phải sử dụng các con số và chỉ số trong dữ liệu để hỗ trợ câu trả lời của mình.
                Dữ liệu tài chính nền tảng:
                {data_for_ai_markdown}
                """
                
                chat_session = get_chat_session(api_key, SYSTEM_PROMPT)

                if chat_session:
                    # Hiển thị lịch sử chat
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Xử lý input từ người dùng
                    if prompt := st.chat_input("Hỏi về Tăng trưởng, Cơ cấu, hoặc Thanh toán..."):
                        # Thêm tin nhắn người dùng vào lịch sử
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Gửi tin nhắn và chờ phản hồi từ Gemini (Đã xử lý lỗi client closed)
                        with st.chat_message("assistant"):
                            with st.spinner("Đang tìm kiếm và phân tích..."):
                                try:
                                    response = chat_session.send_message(prompt)
                                    st.markdown(response.text)
                                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                                except Exception as e:
                                    # BẮT LỖI GỬI TIN NHẮN 
                                    error_msg = f"Lỗi gửi tin nhắn: Vui lòng refresh trang và thử lại. Chi tiết lỗi: {e}"
                                    st.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


        # Xử lý lỗi tổng quát (đã tối ưu hóa)
        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}. Vui lòng kiểm tra file Excel có đủ chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc 3 cột không.")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
            # Đảm bảo xóa chat client để khởi tạo lại
            if "chat_client" in st.session_state:
                del st.session_state["chat_client"] 
                st.session_state["messages"] = []

else:
    # Reset chat session khi không có file
    if "chat_client" in st.session_state:
        del st.session_state["chat_client"]
        st.session_state["messages"] = []
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
