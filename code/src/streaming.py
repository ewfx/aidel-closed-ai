import streamlit as st
import requests

st.set_page_config(page_title="Transaction Validator", page_icon="⚠️", layout="centered")

# Backend API URL (Update with actual server address if hosted)
BACKEND_URL = "http://127.0.0.1:8000/upload"  # FastAPI backend URL

# Custom styled title
st.markdown(
    """
    <h1 style='text-align: center; color: #0077b6; font-size: 36px; font-weight: bold;'>
        Fraud Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

# File uploader (accepts multiple files)
uploaded_files = st.file_uploader("Choose files (CSV or TXT)", type=["csv", "txt"], accept_multiple_files=True)

# Submit button
if st.button("Submit"):
    if uploaded_files:
        # st.info("📤 Sending files to backend...")

        # Prepare files for API request
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]

        try:
            # Send files to backend API
            response = requests.post(BACKEND_URL, files=files)

            # Check if request was successful
            if response.status_code == 200:
                st.success(f"✅ {len(uploaded_files)} File(s) uploaded successfully!")
                st.json(response.json())  # Display response from backend
            else:
                st.error(f"❌ Upload failed! Backend responded with: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to backend: {e}")

    else:
        st.error("⚠️ Please upload at least one file before submitting.")