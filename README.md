# ProcessUnravel Pro 🚀

A system that scrapes, parses, and maps regulatory/government processes into interactive, personalized decision trees using AI-powered agents.

## 🧠 Core Innovation

ProcessUnravel Pro transforms static government processes into dynamic, personalized decision trees that adapt to user context (location, business type, etc.).

### Key Features:
- **Multi-source scraping**: Government websites, PDFs, FAQs
- **AI-powered extraction**: Agentic step classification and dependency mapping
- **Interactive visualization**: React Flow-based decision trees
- **Personalized paths**: Context-aware filtering based on user attributes

## 🏗️ Architecture

```
├── backend/           # FastAPI backend with AI agents
│   ├── agents/       # LangChain agents for processing
│   ├── scrapers/     # Web scraping and PDF parsing
│   ├── graph/        # NetworkX graph engine
│   └── api/          # REST API endpoints
├── frontend/         # React + React Flow UI
│   ├── components/   # Reusable UI components
│   ├── pages/        # Application pages
│   └── hooks/        # Custom React hooks
└── data/            # Sample data and templates
```

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🧪 Sample Use Case

**User Input**: "I want to start a food truck in Toronto as a sole proprietor"

**System Output**: Personalized decision tree with steps like:
1. Choose business structure → sole proprietor
2. Register business name with Ontario
3. Apply for BIN with CRA
4. Get food handler certification
5. Apply for City of Toronto mobile vending license
6. Health inspection
7. Fire department approval
8. Operating permit → GO

Each node includes description, links, costs, required documents, and dependencies.

## 🛠️ Tech Stack

- **Backend**: FastAPI, LangChain, spaCy, NetworkX
- **Scraping**: Playwright, BeautifulSoup, pdfplumber
- **Frontend**: React, React Flow, TypeScript
- **AI**: Lightweight open-source models, rule-based classification
- **Deployment**: Vercel (frontend), Render (backend)
