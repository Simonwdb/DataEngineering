# ✈ NYC 2023 Flight Dashboard

This project is an interactive Streamlit dashboard for analyzing flight data from and to NYC in 2023. It provides insightful visualizations on delays, airline performance, aircraft types, weather impacts, and more.


## 📁 Project Structure

```
.
├── Data/
│   ├── airports.csv
│   ├── flights_database.db
│   └── flights_part12.pdf
├── main.py                  # Main Streamlit app
├── tasks1.py                # Data preprocessing & delay calculations
├── tasks3.py                # Visualizations (e.g., delay vs weather)
├── tasks4.py                # Additional route & aircraft insights
├── utilities.py             # Database class and helper methods
├── requirements.txt         # Required Python packages
└── README.md
```

## 🚀 Getting Started
### 1. Clone the repo

```bash
git clone https://github.com/Simonwdb/DataEngineering
cd DataEngineering
```


### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```


### 3. Run the Streamlit dashboard

```bash
streamlit run main.py
```


## 📊 Dashboard Features

| Section                         | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| **Overview**                    | General flight stats, top destinations, and airport timezone insights.      |
| **Departure/Arrival Comparison**| Compare delays, volume, and taxi time for selected airports.                |
| **Departure-Arrival Analysis**  | In-depth view of a specific route: volume, delays, aircrafts, airlines.     |
| **Delays & Causes**             | Correlation between distance & delay, visibility conditions, and top delays.|
| **Daily Flights**               | View real flights for a selected date with stats and map visualization.     |
| **Aircraft Types & Speed**      | Visualize speed distribution by aircraft models.                            |
| **Weather Impact**              | Study how wind direction impacts air time using inner product vectors.      |


## 🗃️ Data Sources

All data is stored in the `flights_database.db` SQLite database, which includes:

- **flights** – Flight-level schedule, delays, airports, aircraft
- **airports** – Location metadata (lat, lon, timezone)
- **airlines** – Carrier codes and full airline names
- **planes** – Aircraft models and manufacturers
- **weather** – Historical weather by airport & time


## 📐 Data Assumptions

While wrangling and transforming the raw data from the database, the following assumptions and cleaning steps were made to ensure consistency and usability:

- **NaN / Missing Values**
  - Rows with critical null values (e.g., departure/arrival times, airport codes) were dropped.
  - Non-critical columns may still contain missing values and are handled at visualization level.

- **Duplicates**
  - Duplicate flight records (based on flight ID and scheduled departure) were removed.

- **Time Columns**
  - Time-related columns were converted to consistent datetime formats.
  - `sched_dep_time` and `sched_arr_time` were converted to timestamps for accurate delay calculation.

- **Delays**
  - Departure and arrival delays were recalculated using scheduled vs. actual times when not directly available.
  - `total_delay` is defined as the sum of arrival and departure delays.
  - Negative delays are treated as early arrivals or early departures, and retained for completeness.

- **Timezone Normalization**
  - Timezones were adjusted using airport timezone metadata to align scheduling.
  - Arrival dates were converted to GMT-5 to ensure cross-airport consistency.

- **Taxi and Block Times**
  - Taxi time and block time were calculated if not available, using arrival/departure timestamps.

- **Weather Matching**
  - Weather data is assumed to be airport-local and timestamp-aligned; nearest weather record is joined per origin airport and flight time.

These assumptions are implemented in the data preprocessing logic, primarily located in `tasks4.py`.


## 🔧 Built With

- [Streamlit](https://streamlit.io/) — UI framework
- [Plotly Express](https://plotly.com/python/plotly-express/) — Charts & maps
- [SQLite3](https://www.sqlite.org/index.html) — Relational database
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Data manipulation


## 👨‍💻 Authors

- Built by Simon de Boer (2668555), Daniël Kuiper (2688995), Levi Thé (2650081), and Pim Goederaad (2744585)
- All Bachelor of Science students at Vrije Universiteit Amsterdam.

---