"""
Setup and verification script for RAG project.

Run this to verify your installation and setup.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("\n📌 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (Need 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📌 Checking dependencies...")
    
    required = [
        'langchain',
        'chromadb',
        'openai',
        'streamlit',
        'sentence_transformers',
        'pypdf',
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True


def check_environment():
    """Check environment variables"""
    print("\n📌 Checking environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'Optional - for OpenAI embeddings/LLM',
    }
    
    all_set = True
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"   ✅ {var} is set")
        else:
            print(f"   ⚠️  {var} not set ({description})")
            all_set = False
    
    if not all_set:
        print("\n   Edit .env file to add your API keys")
        print("   cp .env.example .env")
    
    return True  # Not critical for basic functionality


def check_directories():
    """Check project structure"""
    print("\n📌 Checking project structure...")
    
    required_dirs = [
        'src/ingestion',
        'src/embeddings',
        'src/vectordb',
        'src/rag_patterns',
        'ui',
        'notebooks',
        'data/sample_documents',
        'config'
    ]
    
    project_root = Path(__file__).parent
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def test_imports():
    """Test importing main modules"""
    print("\n📌 Testing module imports...")
    
    try:
        from src.ingestion.loaders.document_loaders import load_document
        print("   ✅ Document loaders")
    except Exception as e:
        print(f"   ❌ Document loaders: {str(e)}")
        return False
    
    try:
        from src.ingestion.chunking.chunking_strategies import chunk_document
        print("   ✅ Chunking strategies")
    except Exception as e:
        print(f"   ❌ Chunking strategies: {str(e)}")
        return False
    
    try:
        from src.embeddings.providers.embedding_providers import EmbeddingFactory
        print("   ✅ Embedding providers")
    except Exception as e:
        print(f"   ❌ Embedding providers: {str(e)}")
        return False
    
    try:
        from src.vectordb.chromadb_client import ChromaDBClient
        print("   ✅ Vector database client")
    except Exception as e:
        print(f"   ❌ Vector database client: {str(e)}")
        return False
    
    return True


def run_basic_test():
    """Run a basic end-to-end test"""
    print("\n📌 Running basic functionality test...")
    
    try:
        from src.ingestion.chunking.chunking_strategies import chunk_document
        
        sample_text = "This is a test. It has multiple sentences. Each should be chunked properly."
        
        chunks = chunk_document(
            sample_text,
            strategy="recursive",
            chunk_size=50,
            metadata={'test': True}
        )
        
        if len(chunks) > 0:
            print(f"   ✅ Created {len(chunks)} chunks")
            return True
        else:
            print("   ❌ No chunks created")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False


def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\n📖 Next Steps:\n")
    
    print("1. Start the Streamlit UI:")
    print("   streamlit run ui/app.py\n")
    
    print("2. Or open the getting started notebook:")
    print("   jupyter notebook notebooks/01_getting_started.ipynb\n")
    
    print("3. Read the documentation:")
    print("   - README.md - Project overview")
    print("   - QUICKSTART.md - Quick start guide")
    print("   - PROJECT_SUMMARY.md - Complete feature list\n")
    
    print("4. Explore the code:")
    print("   - src/ingestion/ - Document processing")
    print("   - src/embeddings/ - Embedding providers")
    print("   - src/rag_patterns/ - RAG implementations\n")
    
    print("Happy learning! 🚀")


def main():
    """Main setup verification"""
    print("="*60)
    print("🤖 RAG Project Setup Verification")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_directories(),
        check_dependencies(),
        check_environment(),
        test_imports(),
        run_basic_test()
    ]
    
    if all(checks):
        print("\n✅ All checks passed!")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
