# QSR Financial Model Calculator

A comprehensive Streamlit dashboard for modeling Quick Service Restaurant (QSR) financial projections over 10 years.

## Features

- **10-Year Financial Projections**: Monthly revenue, costs, and profit forecasting
- **Ramp Growth Modeling**: Linear ramp from start to target orders, then post-ramp growth
- **Real Rent & Labor Modeling**: Fixed monthly rent with inflation, labor with staffing floor
- **Investor View**: Profit splits and metrics per investor
- **Sensitivity Analysis**: Order-level sensitivity curves
- **Interactive Dashboard**: Adjustable parameters via sidebar controls

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd QSR-Financial-Model-Calculator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run qsr_streamlit_dashboard.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

1. Push your code to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository: `QSR-Financial-Model-Calculator`
6. Set the main file path: `qsr_streamlit_dashboard.py`
7. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

### Option 2: Heroku

1. Install Heroku CLI
2. Create a `Procfile`:
```
web: streamlit run qsr_streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "qsr_streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t qsr-calculator .
docker run -p 8501:8501 qsr-calculator
```

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Usage

1. Adjust parameters in the sidebar (orders, ticket price, costs, etc.)
2. View projections in the main dashboard
3. Explore different tabs for detailed views and sensitivity analysis

## License

MIT License

