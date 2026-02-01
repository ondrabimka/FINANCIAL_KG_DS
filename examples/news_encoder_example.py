"""
Example: Using different news encoders for financial knowledge graph

This script demonstrates how to use different news encoders:
1. Sentiment Analysis (FinBERT)
2. OpenAI Embeddings
3. One-Hot Encoding
"""

import os
import pandas as pd
from dotenv import load_dotenv
from financial_kg_ds.datasets.encoders import (
    SentimentAnalysisEncoder, 
    OpenAIEmbeddingEncoder,
    OneHotEncoder
)

# Load environment variables
load_dotenv()

# Sample news titles
news_titles = pd.DataFrame({
    'title': [
        'Stock market rallies on strong earnings report',
        'Federal Reserve raises interest rates',
        'Tech giant announces layoffs',
        'Oil prices surge amid supply concerns',
        'Company beats quarterly earnings expectations'
    ]
})

print("=" * 80)
print("News Encoder Comparison")
print("=" * 80)
print(f"\nSample news titles ({len(news_titles)} items):")
for i, title in enumerate(news_titles['title'], 1):
    print(f"  {i}. {title}")

# ============================================================================
# 1. Sentiment Analysis Encoder (FinBERT)
# ============================================================================
print("\n" + "=" * 80)
print("1. Sentiment Analysis Encoder (FinBERT)")
print("=" * 80)

sentiment_encoder = SentimentAnalysisEncoder()
sentiment_embeddings = sentiment_encoder(news_titles['title'])

print(f"\nOutput shape: {sentiment_embeddings.shape}")
print(f"Output dimensions: {sentiment_embeddings.shape[1]} (sentiment label, confidence)")
print(f"\nFirst 3 embeddings:")
for i in range(min(3, len(sentiment_embeddings))):
    sentiment_val, confidence = sentiment_embeddings[i]
    sentiment_label = "Positive" if sentiment_val > 0 else ("Negative" if sentiment_val < 0 else "Neutral")
    print(f"  Title {i+1}: {sentiment_label} (score: {sentiment_val:.1f}, confidence: {confidence:.3f})")

# ============================================================================
# 2. OpenAI Embeddings Encoder
# ============================================================================
print("\n" + "=" * 80)
print("2. OpenAI Embeddings Encoder")
print("=" * 80)

if os.getenv("OPENAI_API_KEY"):
    try:
        # Using small model
        print("\n--- Using text-embedding-3-small ---")
        openai_encoder_small = OpenAIEmbeddingEncoder(
            model="text-embedding-3-small"
        )
        openai_embeddings_small = openai_encoder_small(news_titles['title'])
        
        print(f"Output shape: {openai_embeddings_small.shape}")
        print(f"Output dimensions: {openai_embeddings_small.shape[1]}")
        print(f"Sample embedding (first 10 dims): {openai_embeddings_small[0, :10].tolist()}")
        
        # Using large model (optional, commented out due to cost)
        # print("\n--- Using text-embedding-3-large ---")
        # openai_encoder_large = OpenAIEmbeddingEncoder(
        #     model="text-embedding-3-large"
        # )
        # openai_embeddings_large = openai_encoder_large(news_titles['title'])
        # print(f"Output shape: {openai_embeddings_large.shape}")
        # print(f"Output dimensions: {openai_embeddings_large.shape[1]}")
        
        print("\n✓ OpenAI embeddings generated successfully")
        print("  (Embeddings are cached for future use)")
        
    except Exception as e:
        print(f"\n✗ Error with OpenAI encoder: {e}")
        print("  Make sure your OPENAI_API_KEY is valid")
else:
    print("\n⚠ OPENAI_API_KEY not found in environment variables")
    print("  Set it to test OpenAI embeddings:")
    print("  export OPENAI_API_KEY='your-api-key'  # Linux/Mac")
    print("  set OPENAI_API_KEY=your-api-key       # Windows")

# ============================================================================
# 3. One-Hot Encoder
# ============================================================================
print("\n" + "=" * 80)
print("3. One-Hot Encoder")
print("=" * 80)

onehot_encoder = OneHotEncoder()
onehot_embeddings = onehot_encoder(news_titles['title'])

print(f"\nOutput shape: {onehot_embeddings.shape}")
print(f"Output dimensions: {onehot_embeddings.shape[1]} (one per unique title)")
print(f"Embedding sparsity: {(onehot_embeddings == 0).sum().item() / onehot_embeddings.numel() * 100:.1f}% zeros")

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "=" * 80)
print("Encoder Comparison Summary")
print("=" * 80)

print(f"""
┌────────────────────────┬─────────────┬──────────────────┬─────────────────┐
│ Encoder Type           │ Dimensions  │ Semantic Rich    │ Dependencies    │
├────────────────────────┼─────────────┼──────────────────┼─────────────────┤
│ Sentiment (FinBERT)    │ 2           │ Low (sentiment)  │ transformers    │
│ OpenAI Embeddings      │ 1536/3072   │ High             │ openai, API key │
│ One-Hot Encoding       │ {onehot_embeddings.shape[1]:<11} │ None             │ None            │
└────────────────────────┴─────────────┴──────────────────┴─────────────────┘

Recommendations:
- Use Sentiment Analysis for: Sentiment-focused trading signals
- Use OpenAI Embeddings for: Rich semantic understanding
- Use One-Hot Encoding for: Baseline comparisons only
""")

# ============================================================================
# Usage with Graph Loader
# ============================================================================
print("=" * 80)
print("Usage with Graph Loader")
print("=" * 80)

print("""
To use these encoders in your training:

1. Via Configuration File (configs/models/base_gnn.yaml):
   
   news_encoder:
     type: "openai"  # or "sentiment", "onehot"
     openai_model: "text-embedding-3-small"

2. Via Command Line:
   
   python -m financial_kg_ds.train.train_node_regression --news-encoder openai
   
3. Programmatically:
   
   from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
   
   data = GraphLoaderRegresion.get_data(
       news_encoder_type="openai",
       news_openai_model="text-embedding-3-small"
   )

See docs/NEWS_ENCODER_GUIDE.md for detailed documentation.
""")

print("=" * 80)
print("Example completed!")
print("=" * 80)
