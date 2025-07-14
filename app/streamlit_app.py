import streamlit as st
import json
import os
from pathlib import Path
import sys

# Add root for import access
sys.path.append(str(Path(__file__).parent.parent))

from model.final_rag_system import (
    search_qdrant, create_enriched_prompt, query_hf_llm,
    infer_language_from_prompt, log_interaction
)

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="GenCodeRAG", layout="wide")

# App title
st.title("🧠 GenCodeRAG: AI Code Generation using GitHub + RAG")

# Input prompt
query = st.text_area(
    "💬 Enter your code generation prompt",
    placeholder="e.g., Create a Python function to hash passwords using bcrypt",
    height=100
)

# Submit button
if st.button("🚀 Generate Code"):
    if query.strip():
        # Auto language detection
        language = infer_language_from_prompt(query)
        with st.spinner("🔍 Retrieving relevant code chunks..."):
            results = search_qdrant(query, language)

        if results:
            # Log and display context
            st.success(f"✅ Retrieved {len(results)} relevant code examples")
            with st.expander("📚 Context Used (Top 3)"):
                for i, r in enumerate(results[:3], 1):
                    payload = r.payload
                    st.markdown(f"**{i}. {payload.get('repo')} / {payload.get('path')}**")
                    st.code(payload.get('code', '')[:300] + '...', language=payload.get("language", "text"))

            # Generate final response
            with st.spinner("🧠 Generating code..."):
                prompt = create_enriched_prompt(query, results)
                response = query_hf_llm(prompt)
                chunk_ids = [r.payload.get("chunk_id") for r in results]
                log_interaction(query, language, response, chunk_ids)

            # Show output
            st.subheader("🧠 Generated Code")
            st.code(response, language=language or "text")
        else:
            st.warning("⚠️ No relevant examples found.")
    else:
        st.error("❌ Please enter a valid prompt.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Qdrant + Hugging Face")
