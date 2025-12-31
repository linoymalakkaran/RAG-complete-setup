# Contributing to RAG Learning Project

Thank you for your interest in contributing! This project is designed to help people learn about RAG systems.

## Ways to Contribute

### 1. Code Contributions

#### Adding New RAG Patterns
- Fork the repository
- Create a new file in `src/rag_patterns/`
- Inherit from `BasicRAG` or create a new base
- Add comprehensive docstrings explaining the pattern
- Update documentation

#### Adding New Embedding Providers
- Add to `src/embeddings/providers/`
- Inherit from `EmbeddingProvider`
- Update `EmbeddingFactory`
- Add tests

#### Adding New Vector Databases
- Add to `src/vectordb/`
- Implement consistent interface with ChromaDB client
- Document setup instructions

### 2. Documentation

- Improve existing documentation
- Add more concept explanations
- Create tutorial notebooks
- Add code examples

### 3. Testing

- Write unit tests
- Add integration tests
- Create test datasets
- Document test cases

### 4. UI/UX

- Improve Streamlit interface
- Add visualizations
- Create new dashboard pages
- Enhance user experience

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/rag.git
cd rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you create this

# Install pre-commit hooks (if available)
pre-commit install
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Add comments for complex logic
- Keep functions focused and small

### Example

```python
def process_document(
    file_path: str,
    chunk_size: int = 1000,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Process document and create chunks.
    
    Args:
        file_path: Path to document file
        chunk_size: Maximum characters per chunk
        metadata: Optional metadata to attach
        
    Returns:
        List of Chunk objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> chunks = process_document("policy.pdf", chunk_size=500)
        >>> len(chunks)
        10
    """
    # Implementation
    pass
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chunking.py

# Run with coverage
pytest --cov=src tests/
```

## Pull Request Process

1. **Create an issue first** describing what you want to add/fix
2. **Fork and create a branch** from `main`
3. **Make your changes**:
   - Add tests
   - Update documentation
   - Ensure tests pass
4. **Submit PR** with clear description
5. **Respond to feedback**

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added unit tests
- [ ] Tested manually
- [ ] All tests pass

## Documentation
- [ ] Updated README if needed
- [ ] Added docstrings
- [ ] Updated relevant docs

## Checklist
- [ ] Code follows project style
- [ ] Comments added for complex logic
- [ ] No breaking changes (or documented)
```

## Areas We'd Love Help With

### High Priority
- [ ] Implement remaining RAG patterns (CRAG, Agentic, Graph, Multimodal)
- [ ] Add RAGAS evaluation integration
- [ ] Create more tutorial notebooks
- [ ] Build evaluation dashboard UI

### Medium Priority
- [ ] Add more embedding providers
- [ ] Implement caching layer
- [ ] Add query enhancement features
- [ ] Create more visualizations

### Nice to Have
- [ ] FastAPI server implementation
- [ ] Monitoring dashboard
- [ ] Multi-language support
- [ ] More example datasets

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation in `docs/`

## Code of Conduct

Be respectful and constructive. This is a learning project - all skill levels are welcome!

## License

By contributing, you agree your contributions will be licensed under the project's MIT License.
