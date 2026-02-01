# Implementation Summary: Custom OpenAI Encoder for News Features

## Overview
Added support for using OpenAI embeddings as an alternative to FinBERT sentiment analysis for encoding news node features in the financial knowledge graph.

## Changes Made

### 1. Core Implementation Files

#### `financial_kg_ds/datasets/encoders.py`
- **Added**: `OpenAIEmbeddingEncoder` class
  - Encodes news text using OpenAI's embedding API
  - Supports both `text-embedding-3-small` (1536-dim) and `text-embedding-3-large` (3072-dim)
  - Includes caching mechanism to avoid redundant API calls
  - Handles errors gracefully with zero-vector fallback
  - Configurable via API key (from env var or parameter)

#### `financial_kg_ds/datasets/graph_loader.py`
- **Updated**: `GraphLoaderRegresion` class
  - Added `news_encoder_type` parameter to `__init__` (default: "sentiment")
  - Added `news_openai_model` parameter to `__init__` (default: "text-embedding-3-small")
  - Modified `add_news_node()` to accept encoder configuration
  - Updated `get_data()` classmethod to pass encoder parameters
  - Updated `load_full_graph()` to use configured encoder

- **Updated**: Import statements to include `OpenAIEmbeddingEncoder`

#### `financial_kg_ds/train/train_node_regression.py`
- **Added**: Command-line arguments for news encoder configuration
  - `--news-encoder`: Choose between sentiment, openai, onehot
  - `--openai-model`: Select OpenAI model variant
  
- **Updated**: Configuration loading
  - Reads `news_encoder` section from model config
  - Overrides with command-line arguments if provided
  - Logs encoder configuration at startup

- **Updated**: Data loading
  - Passes encoder configuration to `GraphLoaderRegresion.get_data()`
  - Applies same configuration in `evaluate_financial_performance()`

### 2. Configuration Files

#### `configs/models/base_gnn.yaml`
- **Added**: `news_encoder` section
  ```yaml
  news_encoder:
    type: "sentiment"  # Options: sentiment, openai, onehot
    openai_model: "text-embedding-3-small"  # For type="openai"
  ```

### 3. Dependencies

#### `requirements.txt`
- **Added**: `openai>=1.0.0`

### 4. Documentation

#### `docs/NEWS_ENCODER_GUIDE.md` (NEW)
Comprehensive guide covering:
- Overview of all three encoder types
- Configuration instructions
- Usage examples (configuration file, command-line, programmatic)
- Performance considerations
- Troubleshooting guide
- Model architecture recommendations
- Use case comparison table

#### `examples/news_encoder_example.py` (NEW)
Interactive example demonstrating:
- How to use each encoder type
- Output format comparisons
- Integration with graph loader
- Setup instructions for OpenAI API

#### `README.md`
- **Updated**: Configuration section with news encoder options
- **Updated**: Training section with OpenAI encoder examples
- **Added**: Environment variable setup instructions

## Features

### Encoder Options

1. **Sentiment Analysis (FinBERT)** - Default
   - 2-dimensional output: (sentiment_label, confidence_score)
   - Fast, local inference
   - No API key required
   - Good for sentiment-driven strategies

2. **OpenAI Embeddings** - New Feature
   - 1536-dim (small) or 3072-dim (large) output
   - Rich semantic understanding
   - Requires OPENAI_API_KEY
   - Cached for efficiency
   - Best for semantic understanding

3. **One-Hot Encoding** - Baseline
   - Sparse categorical encoding
   - No external dependencies
   - High dimensionality
   - Baseline comparisons only

### Key Capabilities

- **Flexible Configuration**: Configure via YAML, command-line, or code
- **Automatic Caching**: OpenAI embeddings are cached to avoid redundant API calls
- **Error Handling**: Graceful fallback to zero vectors on API errors
- **Backward Compatible**: Default behavior unchanged (uses sentiment encoder)

## Usage Examples

### Via Configuration File
```yaml
# configs/models/base_gnn.yaml
news_encoder:
  type: "openai"
  openai_model: "text-embedding-3-small"
```

```bash
python -m financial_kg_ds.train.train_node_regression
```

### Via Command Line
```bash
export OPENAI_API_KEY='your-api-key'
python -m financial_kg_ds.train.train_node_regression \
  --news-encoder openai \
  --openai-model text-embedding-3-small
```

### Programmatically
```python
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
import os

os.environ['OPENAI_API_KEY'] = 'your-api-key'

data = GraphLoaderRegresion.get_data(
    news_encoder_type="openai",
    news_openai_model="text-embedding-3-small"
)
```

## Testing Recommendations

1. **Test with Sentiment Encoder** (default)
   ```bash
   python -m financial_kg_ds.train.train_node_regression --quick
   ```

2. **Test with OpenAI Embeddings**
   ```bash
   export OPENAI_API_KEY='your-key'
   python -m financial_kg_ds.train.train_node_regression \
     --news-encoder openai \
     --quick
   ```

3. **Run Example Script**
   ```bash
   python examples/news_encoder_example.py
   ```

## Architecture Considerations

When using OpenAI embeddings (1536 or 3072 dimensions):
- Consider increasing `hidden_channels` in model config
- May need more training epochs for convergence
- GNN needs to learn compression of high-dimensional features

Suggested adjustments for OpenAI embeddings:
```yaml
model:
  optuna_params:
    hidden_channels:
      min: 128    # Increased from 64
      max: 1024   # Increased from 512
```

## Cost Considerations

**OpenAI API Pricing** (as of implementation):
- `text-embedding-3-small`: ~$0.02 per 1M tokens
- `text-embedding-3-large`: ~$0.13 per 1M tokens

**Cost Mitigation:**
- Automatic caching prevents repeated API calls
- Start with small model for prototyping
- Cache directory: `financial_kg_ds/data/openai_embeddings_cache`

## Environment Variables Required

For OpenAI embeddings:
```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional (existing)
TRAIN_DATA_PATH=path/to/train/data
EVAL_DATA_PATH=path/to/eval/data
TEST_DATA_PATH=path/to/test/data
```

## Validation Checklist

- [x] OpenAI encoder class implemented with caching
- [x] Graph loader updated to support encoder configuration
- [x] Training script updated with CLI arguments
- [x] Configuration files updated
- [x] Dependencies added (openai)
- [x] Comprehensive documentation written
- [x] Example script created
- [x] README updated with new feature
- [x] Backward compatibility maintained (default: sentiment)

## Future Enhancements

Potential improvements:
1. Add more embedding providers (Cohere, Anthropic, etc.)
2. Support for multiple news encoders simultaneously
3. Ensemble encoding (combine sentiment + embeddings)
4. Automatic optimal encoder selection via hyperparameter tuning
5. Batch processing for OpenAI API efficiency
6. Local embedding models (e.g., sentence-transformers)

## Files Modified/Created

### Modified
- `financial_kg_ds/datasets/encoders.py`
- `financial_kg_ds/datasets/graph_loader.py`
- `financial_kg_ds/train/train_node_regression.py`
- `configs/models/base_gnn.yaml`
- `requirements.txt`
- `README.md`

### Created
- `docs/NEWS_ENCODER_GUIDE.md`
- `examples/news_encoder_example.py`
- `IMPLEMENTATION_SUMMARY.md` (this file)

## Conclusion

The implementation successfully adds flexible news encoding options while maintaining backward compatibility. Users can now choose between:
- Fast sentiment analysis (default)
- Rich OpenAI embeddings (new)
- Simple one-hot encoding (baseline)

All three options are fully integrated with the training pipeline and configurable via YAML, CLI, or code.
