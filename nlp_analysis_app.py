import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json

class NLPAnalyzer:
    def __init__(self, paragraph):
        self.paragraph = paragraph
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
    
    def rankbrain_simulate(self):
        tokens = self.tokenizer.encode(self.paragraph, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            hidden_states = outputs.last_hidden_state
        
        semantic_vectors = hidden_states.mean(dim=1).numpy()
        
        return {
            'token_count': len(tokens),
            'semantic_density': float(np.linalg.norm(semantic_vectors)),
            'unique_tokens_ratio': len(set(tokens)) / len(tokens)
        }
    
    def bert_contextual_analysis(self):
        marked_text = "[CLS] " + self.paragraph + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs.last_hidden_state
        
        token_embeddings = hidden_states[0]
        
        return {
            'total_tokens': len(tokenized_text),
            'embedding_dimensions': token_embeddings.shape[1],
            'embedding_variance': float(torch.var(token_embeddings))
        }
    
    def comparative_analysis(self):
        rankbrain_results = self.rankbrain_simulate()
        bert_results = self.bert_contextual_analysis()
        
        return {
            'rankbrain_metrics': rankbrain_results,
            'bert_metrics': bert_results,
            'semantic_complexity_score': (
                rankbrain_results['semantic_density'] * 
                bert_results['embedding_variance']
            )
        }

def main():
    st.title('RankBrain and BERT Text Analysis')
    
    # Sidebar for input
    st.sidebar.header('Input Paragraph')
    paragraph = st.sidebar.text_area(
        'Enter your text here', 
        'Machine learning transforms how we understand and process language.'
    )
    
    # Analysis button
    if st.sidebar.button('Analyze Text'):
        try:
            # Perform analysis
            analyzer = NLPAnalyzer(paragraph)
            results = analyzer.comparative_analysis()
            
            # Display results
            st.header('Analysis Results')
            
            # RankBrain Metrics
            st.subheader('RankBrain Metrics')
            st.json(results['rankbrain_metrics'])
            
            # BERT Metrics
            st.subheader('BERT Metrics')
            st.json(results['bert_metrics'])
            
            # Semantic Complexity
            st.subheader('Semantic Complexity Score')
            st.write(f"{results['semantic_complexity_score']:.4f}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
