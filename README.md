Certainly! Here's your project README.md formatted properly with Markdown for clarity and readability:

# InsightInvest

**InsightInvest** is an AI-powered chatbot designed to democratize financial intelligence. It acts as a personal investment research analyst, performing deep, multi-faceted research on any publicly traded company to generate a comprehensive investment outlook report.

✨ [View the Live Demo](#) ✨

---

## 🚀 Core Features

- **🤖 Conversational Interface:**  
  Simply enter a company name or stock ticker (e.g., "Nvidia", "AAPL", "RELIANCE.NS") to begin the analysis.

- **🌐 Dynamic Web Research:**  
  Gathers real-time news articles and press releases from Google News to gauge market perception.

- **📊 Quantitative Financial Analysis:**  
  Integrates with the Yahoo Finance API to fetch and interpret key financial metrics like P/E Ratio, EPS, Debt-to-Equity, and revenue trends.

- **🧠 Advanced Sentiment Analysis:**  
  Uses the FinBERT model, specialized for financial text, to perform nuanced sentiment analysis on news headlines, assessing if the market sentiment is positive, negative, or neutral.

- **🔮 AI-Powered Predictive Modeling:**  
  Employs a sophisticated hybrid forecasting model (ARIMA + Holt-Winters) and fuses it with the news sentiment score to predict potential future stock price movements, complete with a visual chart and confidence intervals.

- **📝 Comprehensive Report Generation:**  
  Leverages Google's Gemini LLM to synthesize all the gathered qualitative and quantitative data into a coherent, professional, and easy-to-read investment report.

---

## 🛠️ Technology Stack & Architecture

The project is a monorepo containing a separate frontend and backend, ensuring a clean separation of concerns.

| Component | Technology                          |
| --------- | --------------------------------- |
| Frontend  | Next.js, React, TypeScript, Tailwind CSS |
| Backend   | Python, FastAPI (API), Gunicorn (production) |
| AI & Data | Google Gemini (Report Generation), Hugging Face FinBERT (Sentiment), yfinance (Financial Data), Statsmodels (Forecasting) |

### System Architecture

```mermaid
graph TD
    A[User] --> B{Next.js Frontend on Vercel}
    B --> C{FastAPI Backend on Render}
    C --> D[Yahoo Finance API]
    C --> E[Google News RSS]
    E --> F[Hugging Face API - FinBERT]
    subgraph "Data Synthesis & Analysis"
        direction LR
        D[Stock Data & Metrics] --> G{Forecasting Model}
        F[Sentiment Score] --> G
        G[Forecast Output] --> H{Gemini LLM}
        D --> H
        F --> H
    end
    H --> C
    C --> B
    B --> A
```

---

## ⚙️ Getting Started: Running Locally

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Git
* Python 3.10+
* Node.js and npm

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/InsightInvest.git
cd InsightInvest
```

### 2. Set Up Environment Variables

The project requires API keys for Google Gemini and Hugging Face.

Create a file named `.env` in the backend by copying the example file:

```bash
cp .env.example .env
```

Open `.env` and add your API keys:

```env
# .env
ALPHA_VANTAGE_API_KEY=alpha_vantage_api_key_here
GEMINI_API_KEY=gemini_api_key_here
HF_API_KEY=hugging_face_api_key_here
```

### 3. Backend Setup (FastAPI)

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The backend API will now be running at `http://127.0.0.1:8000`.

### 4. Frontend Setup (Next.js)

Open a **new terminal window** and navigate to the root of the project.

```bash
# Navigate to the frontend directory
cd investchat

# Install Node.js dependencies
npm install

# Create a local environment file for the frontend
cp ../.env.example .env.local

# Edit .env.local and set the API URL for local development
# NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"

# Start the frontend development server
npm run dev
```

The frontend application will now be running at `http://localhost:3000`.

---

## 🌐 API Endpoint

### Get Comprehensive Analysis

* **URL:** `/forecast/{symbol}`

* **Method:** GET

* **URL Params:**
  `symbol=[string]` (e.g., AAPL, TSLA)

* **Query Params:**

  * `steps=[integer]` (optional, default: 10): Number of periods to forecast.
  * `period=[string]` (optional, default: 6mo): Historical data period.
  * `news_items=[integer]` (optional, default: 15): Number of news articles to analyze.

* **Success Response:**

  * **Code:** 200
  * **Content:** A JSON object containing the company info, financial metrics, sentiment analysis, price forecast, a base64-encoded chart image, and the full AI-generated report.

---

## 🚀 Deployment

This application is designed for easy deployment on modern cloud platforms.

* **Backend (FastAPI):**
  Deployed on Render as a Python Web Service.
  Start command:

  ```bash
  gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
  ```

* **Frontend (Next.js):**
  Deployed on Vercel.
  The `NEXT_PUBLIC_API_URL` environment variable is set to the public URL of the deployed Render backend.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
