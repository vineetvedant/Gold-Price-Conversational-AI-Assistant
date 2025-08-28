# 🏆 Gold Price Conversational AI Assistant

A sophisticated AI-powered conversational assistant for gold price analysis, built with modern machine learning and vector search technologies. This project combines FastAPI, ChromaDB, Ollama, and DeepSeek R1 to provide intelligent gold price insights and digital gold purchasing capabilities.

## 📋 Table of Contents

- [🏆 Gold Price Conversational AI Assistant](#-gold-price-conversational-ai-assistant)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🏗️ Architecture Overview](#️-architecture-overview)
  - [🛠️ Technology Stack](#️-technology-stack)
  - [📊 Data Structure](#-data-structure)
  - [🚀 Quick Start](#-quick-start)
  - [📖 Detailed Installation](#-detailed-installation)
  - [⚙️ Configuration](#️-configuration)
  - [🔧 API Documentation](#-api-documentation)
  - [🤖 AI/ML Components](#-aiml-components)
  - [💾 Database & Storage](#-database--storage)
  - [📈 Usage Examples](#-usage-examples)
  - [🔒 Security Considerations](#-security-considerations)
  - [🚀 Deployment](#-deployment)
  - [🔧 Development](#-development)
  - [📊 Monitoring & Analytics](#-monitoring--analytics)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## ✨ Features

### 🤖 Intelligent Conversational AI
- **Natural Language Processing**: Advanced query understanding and classification
- **Context-Aware Responses**: RAG (Retrieval-Augmented Generation) powered by ChromaDB
- **Multi-Modal Queries**: Handles price inquiries, trend analysis, and predictions
- **Purchase Intent Detection**: Automatically detects and routes purchase requests

### 📊 Gold Price Analytics
- **Historical Data Analysis**: Comprehensive analysis of gold price data from 2015+
- **Trend Analysis**: 30-day moving averages and performance metrics
- **Best Time Analysis**: Identifies optimal purchasing periods
- **Real-Time Pricing**: Live price integration with fallback mechanisms

### 🛒 Digital Gold Purchasing
- **Seamless Transactions**: Complete purchase workflow with transaction tracking
- **Flexible Pricing**: Supports multiple price units (per gram, per 10g)
- **Idempotent Operations**: Prevents duplicate transactions
- **Comprehensive Logging**: Detailed transaction history and audit trails

### 🔧 Enterprise-Grade Features
- **Microservices Architecture**: Separated chat and purchase APIs
- **Vector Search**: Fast similarity search using ChromaDB
- **Environment Configuration**: Flexible deployment with environment variables
- **Health Monitoring**: Built-in health checks and status endpoints
- **Thread-Safe Operations**: Concurrent request handling with proper locking

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend/UI   │────│  Chat API       │────│   Ollama +      │
│                 │    │  (Port 8001)    │    │   DeepSeek R1   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Purchase API    │    │   ChromaDB       │    │   CSV Files     │
│ (Port 8002)     │    │   Vector Store   │    │   (Data)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Components

1. **Chat API Service** (`app/main.py`)
   - Handles conversational interactions
   - Routes to purchase service when intent detected
   - Provides health monitoring

2. **Purchase API Service** (`app/purchase.py`)
   - Manages gold purchase transactions
   - Calculates pricing and quantities
   - Maintains transaction records

3. **AI/ML Engine** (`app/model.py`)
   - Query classification and processing
   - RAG implementation with ChromaDB
   - Integration with Ollama/DeepSeek R1

4. **Data Processing** (`app/utils.py`)
   - CSV data loading and preprocessing
   - Price formatting and calculations
   - Statistical analysis functions

## 🛠️ Technology Stack

### Backend Framework
- **FastAPI**: High-performance async web framework
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation and serialization

### AI/ML Stack
- **Ollama**: Local LLM inference server
- **DeepSeek R1**: Advanced reasoning model (1.5B parameters)
- **ChromaDB**: Vector database for RAG
- **ONNX Runtime**: Optimized inference for embeddings
- **sentence-transformers**: Local embedding generation

### Data Processing
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **CSV**: Transaction and price data storage

### Infrastructure
- **Docker**: Containerization (optional)
- **Python 3.8+**: Core runtime
- **Threading**: Concurrent request handling
- **UUID**: Transaction ID generation

## 📊 Data Structure

### Gold Price Data Format
```csv
Date,Price,Open,High,Low,Volume,Chg%
2025-01-06,77149,77309,77542,76545,27160,0.44
2025-01-03,76813,77246,78600,76613,60,-0.05
```

**Columns:**
- `Date`: Trading date (YYYY-MM-DD format)
- `Price`: Closing price per 10 grams (INR)
- `Open`: Opening price per 10 grams (INR)
- `High`: Daily high price per 10 grams (INR)
- `Low`: Daily low price per 10 grams (INR)
- `Volume`: Trading volume
- `Chg%`: Daily percentage change

### Transaction Data Format
```csv
transaction_id,user_id,purchase_amount_inr,gold_amount_grams,price_per_gram,timestamp,status
cbb66c6a-227f-4cbb-a6db-84df901f6558,user_123,15000.00,0.1944,77149.00,2025-08-28T11:53:05.807990+00:00,completed
```

**Columns:**
- `transaction_id`: UUID for unique transaction identification
- `user_id`: User identifier
- `purchase_amount_inr`: Amount spent in Indian Rupees
- `gold_amount_grams`: Gold quantity purchased
- `price_per_gram`: Price per gram at time of purchase
- `timestamp`: ISO 8601 UTC timestamp
- `status`: Transaction status (completed/pending/failed)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- DeepSeek R1 model downloaded

### 1. Clone and Setup
```bash
git clone <repository-url>
cd project_gold
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Ollama and Download Model
```bash
# Start Ollama service
ollama serve

# Download DeepSeek R1 model (in another terminal)
ollama pull deepseek-r1:1.5b
```

### 3. Run the Services
```bash
# Terminal 1: Start Chat API
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Start Purchase API
uvicorn app.purchase:app --host 0.0.0.0 --port 8002 --reload
```

### 4. Test the APIs
```bash
# Health check
curl http://localhost:8001/health
curl http://localhost:8002/health

# Chat with the AI
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current gold price?"}'
```

## 📖 Detailed Installation

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL recommended)
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **Network**: Internet connection for model downloads

### Step-by-Step Installation

#### 1. Install Python Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installations
python -c "import fastapi, chromadb, pandas, httpx; print('All dependencies installed successfully')"
```

#### 2. Install and Configure Ollama
```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download required model
ollama pull deepseek-r1:1.5b

# Verify model installation
ollama list
```

#### 3. Data Setup
```bash
# Ensure data files are in correct location
ls -la app/data/
# Should contain: Gold Price.csv, transactions.csv

# Verify data integrity
python -c "import pandas as pd; df = pd.read_csv('app/data/Gold Price.csv'); print(f'Data loaded: {len(df)} rows')"
```

#### 4. Vector Store Initialization
The vector store is automatically initialized when the chat service starts. The system will:
- Load gold price data from CSV
- Generate embeddings using local ONNX models
- Create ChromaDB collection for RAG

### Troubleshooting Installation

#### Common Issues:

**Ollama Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

**Missing Data Files:**
```bash
# Recreate data directory structure
mkdir -p app/data
cp "Gold Price.csv" app/data/
cp transactions.csv app/data/
```

**Port Conflicts:**
```bash
# Check port availability
netstat -tlnp | grep :8001
netstat -tlnp | grep :8002

# Use different ports
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Chat API Configuration
OLLAMA_BASE_URL=http://localhost:11434
CHAT_MODEL=deepseek-r1:1.5b
PERSIST_DIR=app/vector_store

# Purchase API Configuration
DEFAULT_PRICE_PER_GRAM=6500.00
GOLD_PRICE_CSV=app/data/Gold Price.csv
TRANSACTIONS_CSV=app/data/transactions.csv
PRICE_UNIT=10g

# Security (Optional)
API_KEY=your-secret-key-here
```

### Configuration Options

#### Price Unit Configuration
```bash
# For prices per gram
PRICE_UNIT=gram

# For prices per 10 grams (Indian market standard)
PRICE_UNIT=10g
```

#### Model Configuration
```bash
# Different model options
CHAT_MODEL=deepseek-r1:1.5b    # Fast, lightweight
CHAT_MODEL=deepseek-r1:7b      # Better quality, slower
CHAT_MODEL=llama2:7b           # Alternative model
```

#### Database Configuration
```bash
# Vector store location
PERSIST_DIR=app/vector_store

# Alternative storage backends
# ChromaDB supports cloud storage options
```

## 🔧 API Documentation

### Chat API (Port 8001)

#### POST `/chat`
Main conversational endpoint for gold price queries.

**Request:**
```json
{
  "question": "What is the best time to buy gold this year?"
}
```

**Response:**
```json
{
  "answer": "Based on historical data analysis, the lowest average monthly gold price was ₹45,250 in March 2020...",
  "next_action": null,
  "next_url": null
}
```

**Purchase Intent Response:**
```json
{
  "answer": "I'd be happy to help you purchase digital gold...",
  "next_action": "purchase",
  "next_url": "http://127.0.0.1:8002/purchase"
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "backend": "ollama",
  "base_url": "http://localhost:11434",
  "chat_model": "deepseek-r1:1.5b",
  "embedding_mode": "local all-MiniLM-L6-v2 (Chroma ONNX)",
  "purchase_endpoint": "http://127.0.0.1:8001/purchase"
}
```

### Purchase API (Port 8002)

#### POST `/purchase`
Execute gold purchase transaction.

**Request:**
```json
{
  "userId": "user_123",
  "amountInr": 10000
}
```

**Headers (Optional):**
```
X-Idempotency-Key: unique-request-id
```

**Response:**
```json
{
  "status": "success",
  "message": "Gold purchase successful!",
  "transactionDetails": {
    "transactionId": "cbb66c6a-227f-4cbb-a6db-84df901f6558",
    "userId": "user_123",
    "amountSpentInr": 10000.00,
    "goldPurchasedGrams": 0.1296,
    "pricePerGram": 77149.00,
    "timestamp": "2025-01-06T10:30:00.000000+00:00"
  }
}
```

#### GET `/price`
Get current gold price information.

**Response:**
```json
{
  "pricePerGram": "77149.00",
  "unit": "gram",
  "source": "app/data/Gold Price.csv",
  "assumedPriceUnit": "10g"
}
```

#### GET `/health`
Purchase service health check.

**Response:**
```json
{
  "ok": true,
  "transactionsCsv": "app/data/transactions.csv",
  "priceCsv": "app/data/Gold Price.csv",
  "priceUnit": "10g"
}
```

## 🤖 AI/ML Components

### Query Classification System

The AI engine automatically classifies user queries into categories:

1. **Specific Price Queries**: "What was gold price in January 2024?"
2. **Best Time Analysis**: "When is the cheapest time to buy gold?"
3. **Trend Analysis**: "How has gold performed this year?"
4. **Predictions**: "What will gold price be next year?"
5. **Generic Gold Queries**: "Tell me about gold prices"
6. **Purchase Intent**: "I want to buy gold"

### RAG (Retrieval-Augmented Generation)

**Vector Search Process:**
1. User query is embedded using local ONNX models
2. Similarity search in ChromaDB retrieves relevant historical data
3. Retrieved context is passed to DeepSeek R1 for response generation
4. Response includes data-driven insights with purchase nudges

**Embedding Configuration:**
- Model: `all-MiniLM-L6-v2` (via ONNX Runtime)
- Dimensions: 384
- Distance Metric: Cosine similarity
- Collection: `gold_price_local_all-MiniLM-L6-v2`

### Local AI Inference

**Ollama Integration:**
- Base URL: `http://localhost:11434`
- Model: DeepSeek R1 1.5B parameters
- Inference: Local, privacy-preserving
- Response Format: JSON with streaming support

**Fallback Mechanisms:**
- Graceful degradation on model unavailability
- Cached responses for common queries
- Default responses for off-topic questions

## 💾 Database & Storage

### ChromaDB Vector Store

**Collection Structure:**
```python
collection = chroma_client.get_or_create_collection(
    name="gold_price_local_all-MiniLM-L6-v2",
    embedding_function=local_embedder
)
```

**Data Storage:**
- Documents: Generated sentence descriptions of price data
- Metadata: Structured price information (date, open, high, low, close, volume, change%)
- IDs: Sequential identifiers for each data point

### CSV File Storage

**Gold Price Data:**
- Location: `app/data/Gold Price.csv`
- Format: OHLCV (Open, High, Low, Close, Volume) with percentage change
- Update Frequency: Daily (historical data)

**Transaction Records:**
- Location: `app/data/transactions.csv`
- Format: Structured transaction data with timestamps
- Features: Append-only, thread-safe writes
- Backup: Automatic CSV header creation

### Data Processing Pipeline

1. **Data Loading**: pandas CSV reader with column normalization
2. **Preprocessing**: Date parsing, numeric conversion, missing value handling
3. **Vectorization**: Sentence generation for each price data point
4. **Embedding**: Local ONNX model for vector generation
5. **Storage**: ChromaDB collection persistence

## 📈 Usage Examples

### Basic Price Queries

```bash
# Current price information
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is today'\''s gold price?"}'

# Historical price lookup
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was gold price in December 2024?"}'
```

### Advanced Analytics

```bash
# Trend analysis
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "How has gold performed over the last 30 days?"}'

# Best time analysis
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is the best month to buy gold?"}'
```

### Purchase Flow

```bash
# 1. Initiate purchase conversation
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "I want to buy 10,000 rupees worth of gold"}'

# 2. Execute purchase (redirected from chat response)
curl -X POST "http://localhost:8002/purchase" \
  -H "Content-Type: application/json" \
  -d '{"userId": "user_123", "amountInr": 10000}'
```

### Integration Examples

#### Python Client
```python
import requests
import json

class GoldAssistantClient:
    def __init__(self, chat_url="http://localhost:8001", purchase_url="http://localhost:8002"):
        self.chat_url = chat_url
        self.purchase_url = purchase_url

    def ask_question(self, question):
        response = requests.post(
            f"{self.chat_url}/chat",
            json={"question": question}
        )
        return response.json()

    def purchase_gold(self, user_id, amount_inr):
        response = requests.post(
            f"{self.purchase_url}/purchase",
            json={"userId": user_id, "amountInr": amount_inr}
        )
        return response.json()

# Usage
client = GoldAssistantClient()
response = client.ask_question("What's the current gold price?")
print(response["answer"])
```

#### JavaScript/Node.js Client
```javascript
const axios = require('axios');

class GoldAssistantAPI {
    constructor(chatBaseURL = 'http://localhost:8001', purchaseBaseURL = 'http://localhost:8002') {
        this.chatClient = axios.create({ baseURL: chatBaseURL });
        this.purchaseClient = axios.create({ baseURL: purchaseBaseURL });
    }

    async askQuestion(question) {
        try {
            const response = await this.chatClient.post('/chat', { question });
            return response.data;
        } catch (error) {
            console.error('Chat API error:', error);
            throw error;
        }
    }

    async purchaseGold(userId, amountInr) {
        try {
            const response = await this.purchaseClient.post('/purchase', {
                userId,
                amountInr
            });
            return response.data;
        } catch (error) {
            console.error('Purchase API error:', error);
            throw error;
        }
    }
}

// Usage
const api = new GoldAssistantAPI();
const result = await api.askQuestion("Best time to buy gold?");
console.log(result.answer);
```

## 🔒 Security Considerations

### API Security
- **Input Validation**: Pydantic models ensure data integrity
- **Rate Limiting**: Implement at reverse proxy level (nginx/traefik)
- **CORS Configuration**: Properly configured for production domains
- **HTTPS Enforcement**: SSL/TLS termination required for production

### Data Protection
- **Local AI Inference**: No external API calls for sensitive conversations
- **Transaction Privacy**: User data stored locally in CSV format
- **Audit Trails**: Complete transaction history with timestamps
- **Data Encryption**: Consider encrypting sensitive CSV files

### Production Security
```bash
# Environment variables for sensitive data
export API_KEY="your-production-api-key"
export DATABASE_ENCRYPTION_KEY="your-encryption-key"

# Network security
# Use reverse proxy with SSL termination
# Implement proper firewall rules
# Regular security updates for all dependencies
```

## 🚀 Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p app/data app/vector_store

# Expose ports
EXPOSE 8001 8002 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start script
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  chat-api:
    build: .
    ports:
      - "8001:8001"
    depends_on:
      ollama:
        condition: service_healthy
    volumes:
      - ./app/data:/app/app/data
      - ./app/vector_store:/app/app/vector_store
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001

  purchase-api:
    build: .
    ports:
      - "8002:8002"
    volumes:
      - ./app/data:/app/app/data
    command: uvicorn app.purchase:app --host 0.0.0.0 --port 8002

volumes:
  ollama_data:
```

### Cloud Deployment Options

#### AWS Deployment
```bash
# EC2 instance with GPU support for better performance
# Use ECS/Fargate for containerized deployment
# Implement load balancing with ALB
# Use RDS or S3 for data persistence
```

#### Google Cloud Platform
```bash
# Cloud Run for serverless deployment
# Vertex AI for enhanced ML capabilities
# Cloud Storage for data persistence
# Cloud Build for CI/CD pipelines
```

#### Production Checklist
- [ ] SSL/TLS certificates configured
- [ ] Domain name and DNS setup
- [ ] Load balancer with health checks
- [ ] Monitoring and alerting configured
- [ ] Backup strategy implemented
- [ ] Security groups/firewall rules
- [ ] Environment-specific configurations
- [ ] Performance optimization (caching, CDN)

## 🔧 Development

### Project Structure
```
project_gold/
├── app/
│   ├── __pycache__/
│   ├── data/
│   │   ├── Gold Price.csv      # Historical price data
│   │   └── transactions.csv    # Purchase transactions
│   ├── vector_store/           # ChromaDB persistence
│   ├── main.py                 # Chat API service
│   ├── model.py                # AI/ML logic & RAG
│   ├── purchase.py             # Purchase API service
│   └── utils.py                # Data processing utilities
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
└── .env                        # Environment configuration
```

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd project_gold

# Create development environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Development tools

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
flake8 .
```

### Testing Strategy

**Unit Tests:**
```python
# tests/test_model.py
def test_query_classification():
    assert classify_query("What is gold price today?") == "generic_gold"
    assert classify_query("Best time to buy gold?") == "best_month"

def test_purchase_intent():
    assert is_purchase_intent("I want to buy gold") == True
    assert is_purchase_intent("What is gold?") == False
```

**Integration Tests:**
```python
# tests/test_api_integration.py
def test_chat_api():
    response = client.post("/chat", json={"question": "Hello"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_purchase_flow():
    # Test complete purchase workflow
    pass
```

**Performance Tests:**
```python
# tests/test_performance.py
def test_vector_search_performance():
    # Benchmark ChromaDB queries
    pass

def test_concurrent_requests():
    # Test concurrent API calls
    pass
```

### Code Quality

**Linting and Formatting:**
```bash
# Run all quality checks
black .                    # Code formatting
flake8 .                   # Linting
mypy .                     # Type checking
pytest --cov=app tests/    # Test coverage
```

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## 📊 Monitoring & Analytics

### Health Monitoring

**Endpoints:**
- `/health` - Service health and configuration
- `/metrics` - Performance metrics (implement with prometheus-client)

**Monitoring Setup:**
```python
# Add to requirements.txt
prometheus-client==0.16.0

# Integration in main.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('request_count', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency', 'Request latency', ['method', 'endpoint'])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    with REQUEST_LATENCY.labels(request.method, request.url.path).time():
        response = await call_next(request)
    return response

@app.get("/metrics")
def metrics():
    return generate_latest()
```

### Logging Configuration

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('app.log', maxBytes=10485760, backupCount=5),
            logging.StreamHandler()
        ]
    )

# In main.py
from logging_config import setup_logging
setup_logging()
```

### Analytics Integration

**Usage Tracking:**
```python
# Track user interactions
@app.post("/chat")
async def chat(request: QueryRequest):
    logger.info(f"Chat query: {request.question}")

    # Track query types for analytics
    query_type = classify_query(request.question)
    # Send to analytics service

    response = get_response(request.question)
    return ChatReply(answer=response)
```

**Performance Metrics:**
- Query response times
- Vector search latency
- Purchase conversion rates
- User engagement metrics

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and add tests**
4. **Run quality checks**: `black . && flake8 . && pytest`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Standards

**Python Style Guide:**
- Follow PEP 8 conventions
- Use type hints for all functions
- Write comprehensive docstrings
- Keep functions focused and single-purpose

**Documentation:**
- Update README for new features
- Add docstrings to all public functions
- Include usage examples for complex features
- Update API documentation for endpoint changes

**Testing:**
- Write unit tests for all new functions
- Maintain >80% test coverage
- Test edge cases and error conditions
- Include integration tests for API changes

### Issue Reporting

**Bug Reports:**
```markdown
## Bug Report

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9]
- Ollama Version: [e.g., 0.1.0]
```

**Feature Requests:**
```markdown
## Feature Request

**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
A clear description of any alternative solutions.

**Additional context**
Add any other context about the feature request here.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Gold Price Conversational AI Assistant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

### Open Source Libraries
- **FastAPI**: Modern web framework for building APIs
- **ChromaDB**: AI-native open-source vector database
- **Ollama**: Local LLM inference server
- **DeepSeek R1**: Advanced reasoning model
- **pandas**: Data analysis and manipulation
- **sentence-transformers**: Text embeddings for NLP

### Data Sources
- Gold price data sourced from reliable financial markets
- Historical price information for analytical purposes
- Real-time price integration capabilities

### Community
- Thanks to the open-source AI and ML communities
- Contributors to FastAPI, ChromaDB, and Ollama projects
- Financial data providers and market analysts

---

**Happy Investing! 🏆💰**

For questions or support, please open an issue on GitHub or contact the development team.
