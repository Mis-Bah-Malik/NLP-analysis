import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

def analyze_text(paragraph):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize the paragraph
    tokens = tokenizer.encode(paragraph, add_special_tokens=True)
    input_ids = torch.tensor([tokens])

    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state

    # Calculate metrics
    metrics = {
        'Total Tokens': len(tokens),
        'Unique Tokens': len(set(tokens)),
        'Embedding Dimension': embeddings.shape[-1],
        'Semantic Complexity': float(torch.mean(embeddings))
    }

    return metrics

def main():
    st.title('Text Semantic Analyzer')
    
    # Text input
    paragraph = st.text_area('Enter your text:', 
                              'Machine learning transforms how we understand language.')
    
    # Analyze button
    if st.button('Analyze'):
        try:
            results = analyze_text(paragraph)
            
            # Display results
            st.header('Analysis Results')
            for metric, value in results.items():
                st.write(f"{metric}: {value}")
        
        except Exception as e:
            st.error(f"Analysis failed: {e}")

if __name__ == '__main__':
    main()
