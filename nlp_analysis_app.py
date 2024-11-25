import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel
import numpy as np

class ParagraphAnalyzer:
    def __init__(self, paragraph):
        self.paragraph = paragraph
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def analyze_paragraph(self):
        # Tokenize the paragraph
        marked_text = "[CLS] " + self.paragraph + " [SEP]"
        tokens = self.tokenizer.tokenize(marked_text)
        
        # Convert tokens to ids
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(indexed_tokens)
        
        # Create tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            hidden_states = outputs.last_hidden_state
        
        # Detailed Analysis
        analysis = {
            'Original Text': self.paragraph,
            'Tokens': tokens,
            'Token Count': len(tokens),
            'Unique Words': len(set(tokens)),
            'Semantic Complexity': {
                'Embedding Dimension': hidden_states.shape[-1],
                'Token Embedding Variance': float(torch.var(hidden_states)),
            },
            'Word Breakdown': self.word_level_analysis(tokens)
        }
        return analysis

    def word_level_analysis(self, tokens):
        word_details = []
        for token in tokens:
            word_details.append({
                'token': token,
                'length': len(token),
                'is_special_token': token in ['[CLS]', '[SEP]']
            })
        return word_details

def main():
    st.title('Paragraph Deep Analysis 🔍')
    
    # Paragraph input
    paragraph = st.text_area(
        'Enter your paragraph:',
        'Machine learning transforms how we understand and process language.',
        height=200
    )
    
    # Analyze button
    if st.button('Analyze Paragraph'):
        try:
            analyzer = ParagraphAnalyzer(paragraph)
            results = analyzer.analyze_paragraph()
            
            # Display results
            st.json(results)
        except Exception as e:
            st.error(f"Analysis Error: {e}")

if __name__ == '__main__':
    main()
