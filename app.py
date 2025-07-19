import streamlit as st
import pandas as pd
import torch
from unsloth import FastLanguageModel
from transformers import pipeline, BitsAndBytesConfig
import io
import time

# === Suppress Torch Dynamo Errors ===
torch._dynamo.config.suppress_errors = True

# === Streamlit Configuration ===
st.set_page_config(
    page_title="NLA Question Processor",
    page_icon="🧮",
    layout="wide"
)

# === LaTeX-Aware System Prompt ===
SYSTEM_PROMPT = r"""You are an expert in numerical linear algebra with deep knowledge.
Respond to each user question with a direct, well-structured answer using LaTeX for all mathematical notation.
Use LaTeX to represent:
- matrices (e.g., $A$),
- vectors (e.g., $\vec{x}$),
- norms (e.g., $\lVert x \rVert_2$),
- operators (e.g., $A^T A$, $\kappa(A)$),
- and complexity notations (e.g., $\mathcal{O}(n^3)$).
Do not include explanations about what you're doing.
Avoid meta-comments, apologies, or summaries.
Only provide the clean final answer using appropriate LaTeX syntax."""

@st.cache_resource
def load_model():
    """Load the fine-tuned model with caching"""
    try:
        model_path = "Dilaksan/NLA"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=8192,
            dtype=None,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_questions(pipe, questions_df):
    """Process questions and generate answers"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_questions = len(questions_df)
    
    for idx, row in questions_df.iterrows():
        qid = row.get("id", f"Q{idx+1}")
        question = row.get("questions", "")
        
        if not question.strip():
            continue
            
        # Update progress
        progress = (idx + 1) / total_questions
        progress_bar.progress(progress)
        status_text.text(f"Processing question {idx + 1}/{total_questions}: ID {qid}")
        
        # Generate answer
        full_prompt = SYSTEM_PROMPT.strip() + "\n\nUser Question: " + question.strip()
        
        try:
            response = pipe(full_prompt, max_new_tokens=16384)
            generated = response[0]["generated_text"]
            answer = generated.replace(full_prompt, "").strip()
        except Exception as e:
            st.warning(f"Error processing question ID {qid}: {str(e)}")
            answer = "Error generating answer"
        
        results.append({
            "id": qid,
            "question": question,
            "answer": answer
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ All questions processed!")
    
    return pd.DataFrame(results)

def main():
    st.title("🧮 Numerical Linear Algebra Question Processor")
    st.markdown("Upload your CSV file with questions and get AI-generated answers using LaTeX formatting.")
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.pipe = None
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🤖 Model Controls")
        
        if st.button("Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model... This may take a few minutes."):
                pipe = load_model()
                if pipe:
                    st.session_state.pipe = pipe
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded!")
                else:
                    st.error("❌ Failed to load model")
        
        # Model status indicator
        if st.session_state.model_loaded:
            st.success("🟢 Model Ready")
        else:
            st.warning("🔴 Model Not Loaded")
        
        st.divider()
        
        # Instructions
        st.header("📖 Instructions")
        st.markdown("""
        1. **Load Model** (one-time setup)
        2. **Upload CSV** with 'id' and 'questions' columns
        3. **Process Questions** to generate answers
        4. **Download Results** as CSV
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Upload & Process", "📊 Results"])
    
    with tab1:
        # File upload section
        st.header("Upload Questions CSV")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should have 'id' and 'questions' columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['id', 'questions']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    st.success(f"✅ Valid CSV with {len(df)} questions")
                    
                    # Preview data
                    with st.expander("📋 Data Preview", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(
                            "🚀 Process All Questions", 
                            type="primary", 
                            disabled=not st.session_state.model_loaded,
                            use_container_width=True
                        ):
                            if not st.session_state.model_loaded:
                                st.error("Please load the model first!")
                            else:
                                results_df = process_questions(st.session_state.pipe, df)
                                st.session_state.results = results_df
                                st.balloons()
                                st.success("🎉 Processing complete! Check the Results tab.")
                            
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with tab2:
        if 'results' in st.session_state and not st.session_state.results.empty:
            results_df = st.session_state.results
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(results_df))
            with col2:
                successful = len(results_df[~results_df['answer'].str.contains('Error', na=False)])
                st.metric("Successful", successful)
            with col3:
                errors = len(results_df) - successful
                st.metric("Errors", errors)
            
            # Download section
            st.header("📥 Download Results")
            
            # Prepare download data
            download_df = results_df[['id', 'answer']].copy()
            csv_buffer = io.StringIO()
            download_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="📥 Download Answers CSV",
                    data=csv_data,
                    file_name="nla_answers.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            
            # Results preview
            st.header("📋 Results Preview")
            
            # Sample results with LaTeX rendering
            for idx, row in results_df.head(5).iterrows():
                with st.expander(f"Question {row['id']}", expanded=False):
                    st.write("**Question:**")
                    st.write(row['question'])
                    st.write("**Answer:**")
                    if '$' in str(row['answer']):
                        try:
                            st.latex(row['answer'])
                        except:
                            st.write(row['answer'])
                    else:
                        st.write(row['answer'])
            
            # Full results table
            if st.checkbox("Show Full Results Table"):
                st.dataframe(results_df, use_container_width=True, height=400)
        
        else:
            st.info("👆 Upload and process questions to see results here")

if __name__ == "__main__":
    main()
