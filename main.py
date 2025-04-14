import gradio as gr
import pandas as pd
import os
import time
import tempfile
import shutil
from openai import OpenAI
from langchain_community.document_loaders import CSVLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def extract_mal_risk_factors(
    api_key, 
    base_url,
    system_prompt, 
    user_prompt, 
    input_file, 
    rag_files,
    temperature=0.6, 
    top_p=0.9, 
    max_tokens=2000,
    output_path="./output"
):
    # Record start time
    start_time = time.perf_counter()
    
    # Initialize API client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Create temporary directory for RAG files
    temp_rag_dir = tempfile.mkdtemp(prefix="temp_rag_")
    
    try:
        # Handle input file - Gradio provides the file path directly
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", None
        
        # Check if the dataframe contains a column with "Note" in its name
        note_columns = [col for col in df.columns if 'note' in col.lower()]
        if not note_columns:
            return "No column with 'note' in its name found in the input file.", None
        
        note_column = note_columns[0]  # Use the first column with "note" in its name
        
        # Save uploaded RAG files to temporary directory
        # For Gradio, we just copy the files which are already saved locally
        for rag_file in rag_files:
            file_name = os.path.basename(rag_file)
            dest_path = os.path.join(temp_rag_dir, file_name)
            shutil.copy(rag_file, dest_path)
        
        # Load RAG data
        loader = DirectoryLoader(
            temp_rag_dir,
            glob='**/*.csv',  # Glob pattern to match all CSV files
            loader_cls=CSVLoader,
            use_multithreading=True
        )
        
        rag_data = loader.load()
        
        if not rag_data:
            return "No RAG data loaded. Please check that your RAG files are valid CSV files.", None
        
        # Create FAISS vector store with tiny embedding model
        db = FAISS.from_documents(rag_data, HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="models",
        ))
        
        progress_text = ""
        
        def extract_mal_risk(note, idx, total):
            nonlocal progress_text
            progress_update = f"Processing {idx+1}/{total}...\n"
            progress_text += progress_update
            
            try:
                retriever = db.as_retriever(search_kwargs={"k": 5})
                context_docs = retriever.invoke(note)
                context = "\n".join([doc.page_content for doc in context_docs])
                
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt.format(context=context)
                    },
                    {
                        "role": "user",
                        "content": user_prompt.format(note=note)
                    }
                ]
                
                # Call API
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content, progress_update
            except Exception as e:
                error_msg = f"Error processing note {idx+1}: {str(e)}\n"
                progress_text += error_msg
                return float('nan'), error_msg
        
        # Process each note and update progress
        results = []
        progress_updates = []
        
        for idx, note in enumerate(df[note_column]):
            result, update = extract_mal_risk(note, idx, len(df))
            results.append(result)
            progress_updates.append(update)
            
            # Yield progress updates every 5 records or at the end
            if (idx + 1) % 5 == 0 or idx == len(df) - 1:
                yield "".join(progress_updates), None
                progress_updates = []
        
        # Add results to dataframe
        df["M-Risk factors"] = results
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(output_path, f"risk_factors_{timestamp}.csv")
        
        # Save results
        df.to_csv(output_file, index=False)
        
        # Calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        final_message = f"Processing complete!\nExecution time: {execution_time:.2f} seconds\nResults saved to: {output_file}"
        
        return final_message, output_file
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, None
    
    finally:
        # Clean up temporary RAG directory
        shutil.rmtree(temp_rag_dir, ignore_errors=True)

# Default prompts
default_system_prompt = """You are an expert in aged care nursing and nutrition. Your task is to analyze clinical notes (e.g., nursing progress notes, doctor notes, nutrition specialist notes) from residents in aged care facilities to identify and extract all malnutrition risk factors mentioned.
You should rely on Label of Malnutrition Risk Factor in this context:
{context}"""

default_user_prompt = """Extract malnutrition risk factors from this nursing note:
{note}
In your answer, you must: Identify Explicitly mentioned risk factors, Contextually implied risk factors, Any additional factors necessary for predicting malnutrition that may not be in the RAG file.
Your answer must be a list of Label of malnutrition risk factors like this: item1, item2, item3
If a note contains no identifiable malnutrition risk factors, return "none". 
Please just generate only answer and no explanation."""

# Create Gradio interface
with gr.Blocks(title="Malnutrition Risk Factor Extraction") as app:
    gr.Markdown("# Malnutrition Risk Factor Extraction App")
    
    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(label="API Key", placeholder="Enter your API key", type="password")
            base_url = gr.Textbox(label="Base URL", value="https://api.deepseek.com", placeholder="API endpoint URL")
            
            system_prompt = gr.Textbox(
                label="System Prompt", 
                placeholder="Enter system prompt", 
                value=default_system_prompt,
                lines=5
            )
            
            user_prompt = gr.Textbox(
                label="User Prompt", 
                placeholder="Enter user prompt", 
                value=default_user_prompt,
                lines=5
            )

        with gr.Column():
            input_file = gr.File(label="Input File (CSV/Excel)")
            rag_files = gr.Files(label="RAG Files (CSV)", file_count="multiple")
            
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature", 
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.6, 
                    step=0.1
                )
                top_p = gr.Slider(
                    label="Top P", 
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.9, 
                    step=0.1
                )
                max_tokens = gr.Slider(
                    label="Max Tokens", 
                    minimum=100, 
                    maximum=10000, 
                    value=2000, 
                    step=100
                )
            
            output_path = gr.Textbox(
                label="Output Directory", 
                placeholder="Enter output directory path", 
                value="./output"
            )
    
    extract_button = gr.Button("Extract Malnutrition Risk Factors")
    
    progress = gr.TextArea(label="Progress", lines=10, interactive=False)
    output_file = gr.File(label="Download Results")
    
    extract_button.click(
        fn=extract_mal_risk_factors,
        inputs=[
            api_key, 
            base_url,
            system_prompt, 
            user_prompt, 
            input_file, 
            rag_files, 
            temperature, 
            top_p, 
            max_tokens,
            output_path
        ],
        outputs=[progress, output_file]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()