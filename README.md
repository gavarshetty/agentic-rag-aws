# Agentic RAG System on AWS

A serverless Retrieval-Augmented Generation (RAG) system built using AWS Bedrock, Lambda, API Gateway, and DynamoDB, with a lightweight Streamlit UI.

This application integrates retrieval, reasoning, tool use, and structured agentic workflows—designed for clarity, maintainability, and production-readiness.

## 🚀 Features

### Streamlit UI
- **Chat interface** for user interactions
- Shows model responses, retrieval sources, and tool calls
- Communicates with the backend via HTTP API
- No backend logic in the UI; purely a presentation layer

### AWS Lambda Backend
Responsible for all core logic:
- Handles API Gateway requests
- Runs the agentic RAG pipeline
- Calls Bedrock for:
  - Knowledge Base retrieval
  - Model inference (Claude / Llama 3.1)
- Implements:
  - Query rewriting
  - Reflection loop
  - Tool orchestration
  - Multi-step reasoning

### Agentic Tools
Tools available to the reasoning agent:
- **Web Search Tool** (sub-Lambda function)
- **Calculator Tool** (Python sandbox)
- **Optional: AWS data query tool** (Athena, S3)

### DynamoDB Memory
Stores:
- Multi-turn conversation context
- Reflection loop state
- Tool logs
- Conversation metadata
- Automatic cleanup using TTL

### Bedrock Knowledge Base
Handles:
- Automated chunking
- Embeddings
- Vector indexing
- High-quality semantic retrieval

### Infrastructure-as-Code (AWS CDK)
CDK provisions:
- API Gateway
- Lambda
- DynamoDB
- IAM roles
- Bedrock access
- Optional VPC
- Outputs for the UI
