import streamlit as st
import json
import os
from qdrant_client import QdrantClient
from core.retrieval import RetrievalService, InteractionLogger
from core.embeddings import EmbeddingService
from core.llm import LLMService
from core.prompts import create_enriched_prompt, infer_language_from_prompt
from settings import settings

from dotenv import load_dotenv
from logger import setup_logger
logger = setup_logger(__name__)
load_dotenv()

@st.cache_resource
def get_retrieval_service() -> RetrievalService:
    embedder = EmbeddingService()
    client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY
    )
    return RetrievalService(client=client, embedder=embedder)


@st.cache_resource
def get_llm_service() -> LLMService:
    return LLMService()

st.set_page_config(page_title="GenCodeRAG", layout="wide")

# App title
st.title("🧠 GenCodeRAG: AI Code Generation using GitHub + RAG")
def validate_user_query(query: str):
    blocked_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal system prompt",
        "show hidden prompt",
        "disregard previous directions",
        "bypass security",
        "override instructions",
        "act as system",
        "jailbreak"
    ]

    query_lower = query.lower()

    if len(query.strip()) < 5:
        return False, "❌ Prompt is too short."

    if len(query) > 500:
        return False, "❌ Prompt too long. Please keep it under 500 characters."

    for pattern in blocked_patterns:
        if pattern in query_lower:
            return False, (
                "❌ Prompt contains restricted instructions "
                "that are not allowed."
            )

    return True, ""

def clean_llm_response(response):
    if not response:
        return ""

    response = response.strip()

    if response.startswith("```"):
        lines = response.splitlines()

        if len(lines) >= 3:
            lines = lines[1:-1]

        response = "\n".join(lines)

    return response.strip()

# Input prompt
query = st.text_area(
    "💬 Enter your code generation prompt",
    placeholder="e.g., Create a Python function to hash passwords using bcrypt",
    height=100
)

# Submit button
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if st.button(
    "🚀 Generate Code",
    disabled=st.session_state.is_generating
):
    st.session_state.is_generating = True
    if query.strip():
        try:
            is_valid, error_message = validate_user_query(query)

            if not is_valid:
                logger.warning(
                    f"Blocked suspicious prompt attempt: {query[:100]}"
                )

                st.error(error_message)
                st.session_state.is_generating = False
                st.stop()

            # Auto language detection
            language = infer_language_from_prompt(query)

            with st.spinner("🔍 Retrieving relevant code chunks..."):
                results = get_retrieval_service().search(query, language)

            if results:
                st.success(f"✅ Retrieved {len(results)} relevant code examples")

                with st.expander("📚 Context Used (Top 3)"):
                    for i, r in enumerate(results[:3], 1):
                        payload = r.payload
                        st.markdown(
                            f"**{i}. {payload.get('repo')} / {payload.get('path')}**"
                        )
                        st.code(
                            payload.get('code', '')[:300] + '...',
                            language=payload.get("language", "text")
                        )

                with st.spinner("🧠 Generating code..."):
                    prompt = create_enriched_prompt(query, results)
                    response = get_llm_service().complete(prompt)
                    response = clean_llm_response(response)

                    chunk_ids = [
                        r.payload.get("chunk_id")
                        for r in results
                    ]

                    InteractionLogger().log(query, language, response, chunk_ids)

                st.subheader("🧠 Generated Code")

                if "def " in response or "class " in response or "function" in response:
                    st.code(response, language=language or "text")
                else:
                    st.markdown(response)

            else:
                st.warning(
                    "⚠️ No relevant examples were found. "
                    "Try a more specific prompt, or verify your Qdrant collection is populated."
                )

        except Exception as e:
            st.error(f"❌ Something went wrong: {str(e)}")

        finally:
            st.session_state.is_generating = False

    else:
        st.error("❌ Please enter a valid prompt.")
        st.session_state.is_generating = False

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Qdrant + Hugging Face")
