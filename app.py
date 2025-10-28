import streamlit as st
from google import genai
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Workshop - مقارنة الإجابات",
    page_icon="🤖",
    layout="wide"
)

# Add RTL support for Arabic
st.markdown("""
<style>
    /* RTL support for the entire app */
    .main, .block-container {
        direction: rtl;
        text-align: right;
    }

    /* RTL for input fields */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }

    /* RTL for text areas */
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }

    /* RTL for markdown and text */
    p, h1, h2, h3, h4, h5, h6, li, span {
        direction: rtl;
        text-align: right;
    }

    /* RTL for info/success/warning boxes */
    .stAlert, .stInfo, .stSuccess, .stWarning {
        direction: rtl;
        text-align: right;
    }

    /* RTL for expanders */
    .streamlit-expanderHeader, .streamlit-expanderContent {
        direction: rtl;
        text-align: right;
    }

    /* RTL for sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    /* Better font for Arabic */
    * {
        font-family: 'Arial', 'Helvetica', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

# Initialize ChromaDB
@st.cache_resource
def get_collection():
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection(name="my_documents_collection")
    return collection

client = get_client()
collection = get_collection()

# Title
st.title("🤖 مقارنة الإجابات: بدون RAG vs مع RAG")
st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ معلومات")
    st.write(f"عدد المستندات المحفوظة: **{collection.count()}**")
    st.markdown("---")
    st.subheader("📚 المستندات المتوفرة")
    if collection.count() > 0:
        docs = collection.get()
        for i, doc in enumerate(docs['documents'], 1):
            with st.expander(f"مستند {i}"):
                st.write(doc)

# Main interface
user_query = st.text_input(
    "❓ اكتب سؤالك هنا:",
    # value="متى تأسست شركة النور ومن هو رئيسها التنفيذي؟",
    # placeholder="مثال: كم أرباح شركة النور؟"
)

if st.button("🔍 احصل على الإجابة", type="primary"):
    if user_query:
        # Create two columns for comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("❌ إجابة بدون RAG")
            st.caption("الإجابة المباشرة من النموذج بدون سياق")

            with st.spinner("جاري التوليد..."):
                response_without_rag = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=user_query
                )
                st.info(response_without_rag.text)

        with col2:
            st.subheader("✅ إجابة مع RAG")
            st.caption("الإجابة بناءً على المستندات المسترجعة")

            with st.spinner("جاري البحث والتوليد..."):
                # Get query embedding
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=user_query,
                )
                query_embedding = result.embeddings[0].values

                # Retrieve relevant documents
                relevant_docs = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3
                )

                # Display retrieved context
                context = "\n".join(relevant_docs['documents'][0])
                with st.expander("📄 السياق المسترجع"):
                    st.write(context)

                # Generate response with RAG
                prompt = f"""أنت مساعد ذكي. أجب على السؤال بناءً على السياق فقط.
إذا لم تتمكن من العثور على الإجابة في السياق المقدم، أجب بـ "المعلومة غير متوفرة في السياق."
السياق:
---
{context}
---
السؤال: {user_query}
الإجابة:"""

                response_with_rag = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=prompt
                )
                st.success(response_with_rag.text)

        # Show comparison insights
        st.markdown("---")
        st.subheader("💡 الفرق بين الإجابتين")
        st.write("""
        - **بدون RAG**: النموذج يجيب بناءً على معرفته العامة فقط (قد تكون غير دقيقة أو قديمة)
        - **مع RAG**: النموذج يجيب بناءً على المستندات المحددة التي لديك (أكثر دقة وموثوقية)
        """)
    else:
        st.warning("⚠️ الرجاء إدخال سؤال أولاً")

# Footer
st.markdown("---")
st.caption("RAG Workshop - Retrieval Augmented Generation")
