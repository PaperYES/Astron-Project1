# Astron Solar Eclipse Comprehensive Analysis (2001–2025)

> A fully automated analysis pipeline for NASA solar eclipse data covering 2001 to 2025, including data cleaning, classification, visualization, cycle detection, historical event matching, and cross-cultural interpretation.

---

## Project Overview

This pipeline automates:

- Data ingestion, cleaning, and classification of eclipse records
- LLM-driven observation guidance and visualization of trends
- Saros cycle detection, historical event matching, and comprehensive export

---

## Features

- Rigorous data validation and normalization
- Comprehensive eclipse type statistics and visualizations
- Automated observation guidance using LLM
- Comparative cross-cultural interpretations
- Saros cycle detection and historical event correlation
- High-quality Excel reports and publication-ready charts

---

## Requirements

- Python 3.8 or later
- Dependencies:
  - pandas, numpy, openpyxl
  - matplotlib
  - openai (custom API endpoint)
  - tenacity

---

## Installation and Configuration

```powershell
# Clone or download the repository
cd Astron

# Install dependencies
pip install -r requirements.txt

# Set the API key (environment variable or prompt)
$Env:DEEPSEEK_API_KEY = "YOUR_API_KEY"
```

---

## Usage

### Script Execution

```powershell
python process_2.py
```

Follow the on-screen prompts to select the analysis period. The process will run end-to-end and save all outputs.

### Notebook Demonstration

Open `Project1.ipynb` in Jupyter, execute all cells sequentially, and observe interactive visual outputs.

---

## Project Structure

```text
Astron/
├── NASA_2001-2025_Solar_Eclipses.xlsx    # Source data
├── process_2.py                           # Core analysis script
├── process_3.py                           # Extended cycle and event analysis
├── process_5.py                           # Cross-cultural interpretation module
├── Project1.ipynb                         # Jupyter Notebook demonstration
├── requirements.txt                       # Dependency manifest
└── eclipse_analysis_results/              # Output directory
    ├── *.xlsx                             # Generated Excel reports
    └── *.png                              # Generated charts
```

---

## Example Output

- `eclipse_frequency_analysis.png`: Annual trend bar chart
- `*_eclipse_distribution.png`: Regional distribution pie charts
- `eclipse_observation_tips.xlsx`: Generated observation tips
- `eclipse_cultural_comparison.txt`: Cross-cultural interpretation
- `full_processed_eclipse_data.xlsx`: Final processed dataset

---

## Contributors

- Jacky Peng
- Chenyue Pan

Contributions via issues and pull requests are welcome.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
