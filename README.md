# Early Warning System (EWS) Enhancements

This project enhances the existing Bank Early Warning System with advanced monitoring and alerting capabilities based on the principles of continuous monitoring, data analysis, risk identification, automated alerts, and intervention/correction.

## New Features

### 1. Enhanced Data Model
- Added new financial and behavioral indicators to the loan input model:
  - Capital Adequacy Ratio
  - Liquidity Coverage Ratio
  - Working Capital Cycle
  - Receivables Age
  - GDP Growth Rate
  - Market Index Year-over-Year
  - Unemployment Rate
  - Inflation Rate

### 2. Automated Alert Generation
- Implemented risk identification thresholds that automatically generate alerts:
  - Poor Capital Adequacy (ratio < 10%)
  - Extended Working Capital Cycle (> 90 days)
  - High EWS Score (> 70% probability of default)
- **NEW**: Automated alerts are now generated for high-risk predictions (> 70% default probability)
- Alerts are stored in the database and can be managed through the API

### 3. Alert Management API
- New endpoints for managing alerts:
  - `GET /api/alerts` - Retrieve all active alerts
  - `POST /api/alerts/{alert_id}/update_status` - Update alert status and add intervention notes

### 4. Risk Management Dashboard
- Enhanced risk management interface with:
  - Real-time alerts display
  - Severity-based color coding
  - Intervention tools for risk managers
  - On-site examination note-taking
  - Alert action management

### 5. Macroeconomic Data Integration
- Added support for tracking macroeconomic indicators:
  - `POST /api/macro_indicators` - Add new macroeconomic data
  - `GET /api/macro_indicators` - Retrieve historical macroeconomic data

## Key Components and Indicators

### Internal Financial Data
- Capital adequacy
- Liquidity
- Profitability
- Loan portfolio quality
- Internal controls

### Customer Financial Behavior
- Monitoring borrower behavior
- Tracking customer financial patterns

### Macroeconomic Data
- GDP growth rates
- Market indicators
- Unemployment rates
- Inflation rates

### Market Indicators
- Stock market trends
- Real estate values
- Other market segments

## Implementation Details

### Continuous Monitoring
The system continuously monitors a wide range of data sources, both internal to the bank and external, such as financial markets and economic indicators.

### Data Analysis
The systems process vast amounts of data in real-time, using algorithms to identify patterns and signals that would be impossible for humans to detect manually.

### Risk Identification
The system flags specific risk signals, such as excessive receivables, extended working capital cycles, or poor capital adequacy, that could indicate potential credit distress.

### Automated Alerts
The systems provide alerts to risk managers, enabling them to act quickly for high risk.

### Intervention and Correction

## API Endpoints

### Prediction and Alerts
- `POST /predict` - Make a loan default prediction (enhanced with alert generation)
- `GET /api/alerts` - Get all active alerts
- `POST /api/alerts/{alert_id}/update_status` - Update alert status and notes

### Macroeconomic Data
- `POST /api/macro_indicators` - Add macroeconomic indicators
- `GET /api/macro_indicators` - Get historical macroeconomic indicators

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Initialize the database:
   ```
   python init_db.py
   ```

3. Start the application:
   ```
   uvicorn main:app --reload
   ```

## Testing

Run the test suite:
```
python test_ews.py
```

## Usage

1. Access the dashboard at http://localhost:8000/
2. Use the Risk Management dashboard to view and manage alerts
3. Submit loan applications through the dashboard to trigger risk predictions
4. High-risk predictions (> 70% default probability) will automatically generate alerts