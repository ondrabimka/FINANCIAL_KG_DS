"""
Quick test to verify the OpenAI encoder implementation
Run this to check if the integration is working correctly
"""

import sys
import os

def test_imports():
    """Test that all necessary imports work"""
    print("Testing imports...")
    try:
        from financial_kg_ds.datasets.encoders import (
            SentimentAnalysisEncoder,
            OpenAIEmbeddingEncoder,
            OneHotEncoder
        )
        print("✓ All encoder imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_encoder_initialization():
    """Test that encoders can be initialized"""
    print("\nTesting encoder initialization...")
    
    try:
        from financial_kg_ds.datasets.encoders import (
            SentimentAnalysisEncoder,
            OneHotEncoder
        )
        
        # Test sentiment encoder
        sentiment_enc = SentimentAnalysisEncoder()
        print("✓ SentimentAnalysisEncoder initialized")
        
        # Test one-hot encoder
        onehot_enc = OneHotEncoder()
        print("✓ OneHotEncoder initialized")
        
        # Test OpenAI encoder (might fail if openai not installed)
        try:
            from financial_kg_ds.datasets.encoders import OpenAIEmbeddingEncoder
            # Don't actually initialize without API key
            print("✓ OpenAIEmbeddingEncoder class available")
        except Exception as e:
            print(f"⚠ OpenAIEmbeddingEncoder: {e}")
            print("  (This is OK if openai package not installed yet)")
        
        return True
    except Exception as e:
        print(f"✗ Encoder initialization error: {e}")
        return False

def test_graph_loader_signature():
    """Test that GraphLoaderRegression accepts new parameters"""
    print("\nTesting GraphLoaderRegresion signature...")
    
    try:
        from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
        import inspect
        
        # Check __init__ signature
        init_sig = inspect.signature(GraphLoaderRegresion.__init__)
        params = list(init_sig.parameters.keys())
        
        assert 'news_encoder_type' in params, "Missing news_encoder_type parameter"
        assert 'news_openai_model' in params, "Missing news_openai_model parameter"
        
        print(f"✓ GraphLoaderRegresion.__init__ parameters: {params}")
        
        # Check get_data signature
        get_data_sig = inspect.signature(GraphLoaderRegresion.get_data)
        get_data_params = list(get_data_sig.parameters.keys())
        
        assert 'news_encoder_type' in get_data_params, "Missing news_encoder_type in get_data"
        assert 'news_openai_model' in get_data_params, "Missing news_openai_model in get_data"
        
        print(f"✓ GraphLoaderRegresion.get_data parameters: {get_data_params}")
        
        return True
    except AssertionError as e:
        print(f"✗ Signature check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking signatures: {e}")
        return False

def test_config_file():
    """Test that config file has news_encoder section"""
    print("\nTesting configuration file...")
    
    try:
        import yaml
        config_path = "configs/models/base_gnn.yaml"
        
        if not os.path.exists(config_path):
            print(f"⚠ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'news_encoder' in config, "Missing news_encoder section"
        assert 'type' in config['news_encoder'], "Missing type in news_encoder"
        assert 'openai_model' in config['news_encoder'], "Missing openai_model in news_encoder"
        
        print(f"✓ Configuration file valid")
        print(f"  Encoder type: {config['news_encoder']['type']}")
        print(f"  OpenAI model: {config['news_encoder']['openai_model']}")
        
        return True
    except AssertionError as e:
        print(f"✗ Config validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False

def test_requirements():
    """Test that openai is in requirements.txt"""
    print("\nTesting requirements.txt...")
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        if 'openai' in requirements:
            print("✓ openai package in requirements.txt")
            return True
        else:
            print("✗ openai package NOT in requirements.txt")
            return False
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False

def main():
    print("=" * 80)
    print("OpenAI Encoder Integration Test")
    print("=" * 80)
    
    tests = [
        test_imports,
        test_encoder_initialization,
        test_graph_loader_signature,
        test_config_file,
        test_requirements
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation looks good.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'")
        print("3. Run example: python examples/news_encoder_example.py")
        print("4. Train with OpenAI: python -m financial_kg_ds.train.train_node_regression --news-encoder openai")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
