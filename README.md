# InsightInvest

**InsightInvest** is an AI-powered chatbot designed to democratize financial intelligence. It acts as a personal investment research analyst, performing deep, multi-faceted research on any publicly traded company to generate a comprehensive investment outlook report.

✨ [View the Live Demo](https://insight-invest.vercel.app/) ✨

---

## 🚀 Core Features

- **🤖 Conversational Interface:**
  Simply enter a stock ticker (e.g., "Nvidia", "AAPL", "RELIANCE.NS") to begin the analysis.

- **🌐 Dynamic Web Research:**
  Gathers real-time news headlines via Google News RSS feeds to gauge immediate market perception without heavy scraping overhead.

- **📊 Quantitative Financial Analysis:**
  Integrates with the Yahoo Finance API to fetch and interpret key financial metrics like P/E Ratio, EPS, Debt-to-Equity, and revenue trends.

- **🧠 Advanced Sentiment Analysis:**
  Uses Groq's high-speed inference (Llama 3) to analyze financial news headlines in parallel, rapidly assessing if the global market sentiment is positive, negative, or neutral.

- **🔮 AI-Powered Predictive Modeling:**
  Employs a sophisticated hybrid forecasting model (ARIMA + Holt-Winters) and fuses it with the news sentiment score to predict potential future stock price movements, complete with a visual chart and confidence intervals.

- **📝 Comprehensive Report Generation:**
  Leverages Groq's Llama 3 LLM to synthesize all gathered qualitative and quantitative data into a coherent, professional, and easy-to-read investment report.

---

## 🛠️ Technology Stack & Architecture

The project is a monorepo containing a separate frontend and backend, ensuring a clean separation of concerns.

| Component | Technology                          |
| --------- | --------------------------------- |
| Frontend  | Next.js, React, TypeScript, Tailwind CSS |
| Backend   | Python, FastAPI (API), Gunicorn (production) |
| AI & Data | Groq LPU (Llama 3) (Report & Sentiment), yfinance (Financial Data), Google News RSS, Statsmodels (Forecasting)

### System Architecture

```mermaid
graph TD
    A[User] --> B{"Next.js Frontend on Vercel"}
    B --> C{"FastAPI Backend on Render"}
    C --> D["Yahoo Finance API"]
    C --> E["Google News RSS"]

    subgraph "Data Synthesis & Analysis"
        direction LR
        D["Stock Data & Metrics"] --> G{"Forecasting Model"}
        E["News Headlines"] --> H{"Groq LPU (Llama 3)"}
        H -->|Sentiment Analysis| I["Sentiment Score"]
        I --> G
        G["Forecast Output"] --> H
        D --> H
    end

    H -->|Investment Report| C
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

The project requires API key for Groq (which handles both sentiment and reporting).

Create a file named `.env` in the backend folder.
Open `.env` and add your API keys:

```env
# .env
GROQ_API_KEY=gsk_your_key_here_1, gsk_your_key_here_2
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

# Edit .env.local and set the API URL for local development
# NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"

# Start the frontend development server
npm run dev
```

The frontend application will now be running at `http://localhost:3000`.

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
