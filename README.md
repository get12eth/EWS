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

   If you prefer to use an existing MySQL database instead of the default SQLite file, set the `DATABASE_URL` environment variable before starting the app. Example using PyMySQL driver (note URL-encoding special characters in your password):

   ```bash
   # Example (replace host, user, password and database name):
   export DATABASE_URL="mysql+pymysql://root:Bant%406963@localhost/lon-default"   # macOS/Linux
   # On Windows PowerShell:
   $env:DATABASE_URL = "mysql+pymysql://root:Bant%406963@localhost/lon-default"
   ```

   Make sure you installed the extra requirement `pymysql` (it's in `requirements.txt`). Then run the DB initialization if needed:

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

Migration verification test

A small integration test verifies the migration from the local SQLite `Loan_table` into a MySQL target set by `DATABASE_URL`.

- The test is `tests/test_migration_verification.py` and is skipped unless `DATABASE_URL` is set.
- To run locally (PowerShell):
  ```powershell
  $env:DATABASE_URL = "mysql+pymysql://root:Bant%406963@127.0.0.1:3306/lon-default"
  .\.venv\Scripts\pytest -q tests/test_migration_verification.py::test_migration_verification
  ```

Migration script notes

The migration helper `scripts/migrate_sqlite_to_mysql.py` now:

- Detects the target primary key name (handles `id`, `ID`, etc.)
- Preserves source IDs when `--preserve-ids` is used (skips conflicting IDs)
- Filters out columns not present on the target before inserting to avoid schema mismatch errors

Usage examples (PowerShell):

- Dry-run:
  ```powershell
  $env:DATABASE_URL = "mysql+pymysql://root:Bant%406963@127.0.0.1:3306/lon-default"
  python scripts/migrate_sqlite_to_mysql.py --dry-run
  ```

- Commit (preserve IDs):
  ```powershell
  $env:DATABASE_URL = "mysql+pymysql://root:Bant%406963@127.0.0.1:3306/lon-default"
  python scripts/migrate_sqlite_to_mysql.py --commit --preserve-ids
  ```

CI

A GitHub Actions workflow `.github/workflows/migration-integration.yml` runs this migration and test on PRs and pushes to the default branch. It mirrors the source schema into a MySQL service and runs the migration to ensure the code remains compatible with schema changes.

## Usage

1. Access the dashboard at http://localhost:8000/
2. Use the Risk Management dashboard to view and manage alerts
3. Submit loan applications through the dashboard to trigger risk predictions
4. High-risk predictions (> 70% default probability) will automatically generate alerts