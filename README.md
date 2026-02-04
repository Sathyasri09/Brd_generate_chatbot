# Professional BRD & Technical Spec Generator

An elite AI-driven workstation designed for Business Analysts and Solutions Architects. This application conducts dynamic interviews to gather requirements and generates board-ready Business Requirement Documents (BRD) and Technical Specification Documents (TSD) in `.docx` format.

## Features
- **Mission-Driven Synthesis**: Uses high-level project goals to intelligently draft comprehensive documents.
- **Strict Incremental Editing**: Updates existing drafts based on feedback without rewriting untouched sections.
- **Dynamic Architecture**: Automatically organizes complex requirements into logical hierarchical subheadings.
- **Professional Formatting**: Generates clean, professional docx files with custom styling (14pt Titles, Centered formatting).
- **Dual AI Engine**: Specialized routing between JSON-optimized interview logic and high-density document generation.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- An API Key (Google Gemini or OpenAI/OpenRouter)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
# For Google Gemini (Default)
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.0-flash-exp

# OR For OpenRouter
# LLM_TYPE=openrouter
# OPENROUTER_API_KEY=your_key
# OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

### 4. Running the App
```bash
chainlit run app.py
```

## Project Structure
- `app.py`: Chainlit UI and message handling.
- `nodes.py`: Core AI logic (Interview, Update, and Document Synthesis).
- `graph.py`: LangGraph workflow routing logic.
- `utils.py`: Word document conversion and file utilities.
- `schema.py`: Pydantic models for structured state management.
- `brd_template.txt`: Baseline document architecture.
