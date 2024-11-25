import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel
import numpy as np

class BlogSemanticAnalyzer:
    def __init__(self, article_text):
        self.article = article_text
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def semantic_interpretation(self):
        # Prepare text for analysis
        marked_text = "[CLS] " + self.article + " [SEP]"
        tokens = self.tokenizer.tokenize(marked_text)
        
        # Convert tokens to tensor
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(indexed_tokens)
        
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            embeddings = outputs.last_hidden_state
        
        # Semantic Analysis Calculations
        semantic_density = float(torch.mean(torch.norm(embeddings, dim=-1)))
        contextual_complexity = float(torch.var(embeddings))
        
        # Generate Explanatory Paragraph
        explanation = self.generate_semantic_explanation(
            tokens, 
            semantic_density, 
            contextual_complexity
        )
        
        return {
            'semantic_explanation': explanation,
            'metrics': {
                'total_tokens': len(tokens),
                'semantic_density': semantic_density,
                'contextual_complexity': contextual_complexity
            }
        }

    def generate_semantic_explanation(self, tokens, semantic_density, complexity):
        # Contextual semantic interpretation
        readability_interpretation = (
            f"The text demonstrates a {'rich' if semantic_density > 0.5 else 'moderate'} semantic structure "
            f"with {len(tokens)} tokens. The semantic density of {semantic_density:.2f} suggests "
            f"{'deep' if semantic_density > 0.7 else 'moderate'} information complexity. "
            f"Contextual variations indicate {'nuanced' if complexity > 0.4 else 'consistent'} narrative flow. "
            "RankBrain and BERT analysis reveal the text's underlying semantic relationships, "
            "capturing subtle contextual meanings beyond simple word-by-word interpretation."
        )
        return readability_interpretation

def main():
    st.title('Blog Article Semantic Analyzer 📝')
    
    # Article input
    article = st.text_area(
        'Paste your blog article:',
        'Machine learning transforms how we understand and process complex language patterns. '
        'Advanced algorithms enable deeper insights into textual semantics and contextual understanding.',
        height=200
    )
    
    # Analyze button
    if st.button('Analyze Semantic Structure'):
        try:
            analyzer = BlogSemanticAnalyzer(article)
            results = analyzer.semantic_interpretation()
            
            # Display semantic explanation
            st.subheader('Semantic Interpretation')
            st.write(results['semantic_explanation'])
            
            # Display detailed metrics
            st.subheader('Semantic Metrics')
            st.json(results['metrics'])
        
        except Exception as e:
            st.error(f"Analysis Error: {e}")

if __name__ == '__main__':
    main()
