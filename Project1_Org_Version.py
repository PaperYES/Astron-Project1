import configparser
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import time
import os
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ConfigManager: handles reading paths, API key, and LLM client setup
class ConfigManager:
    def __init__(self):
        # Basic path configuration
        self.excel_path = "NASA_2001-2025_Solar_Eclipses.xlsx"
        self.output_dir = "eclipse_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        # LLM configuration
        self.api_key = os.environ.get("DEEPSEEK_API_KEY") or configparser.ConfigParser().get('API', 'KEY', fallback=None) or input("Enter API Key: ")
        # Validate API key
        if not self.api_key:
            print("Error: API key is missing or invalid. Please set DEEPSEEK_API_KEY environment variable.")
            sys.exit(1)
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key, base_url="https://babeltower.pro/v1", timeout=60)
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client: {e}")
            sys.exit(1)
        self.model_name = "deepseek-r1"
        # Business configuration
        self.historical_events = {
            "2009-07-22": "The longest total solar eclipse occurred in China's Yangtze River Basin, observed by over 10 million people.",
            "2017-08-21": "Great American Eclipse: Total eclipse path covered the entire American continent, sparking a nationwide observation craze.",
            "2020-06-21": "Complete annular solar eclipse visible in the Ali Region.",
            "2023-04-20": "Hybrid solar eclipse experienced in Indonesia and Australia; partial eclipse visible in Hainan, China."
        }
        self.classification_dict = {
            "Total Eclipse (Total)": ["Total"],
            "Partial Eclipse (Partial)": ["Partial"],
            "Annular Eclipse (Annular)": ["Annular"],
            "Hybrid Eclipse (Hybrid)": ["Hybrid"]
        }
        self.color_map = {
            "Total Eclipse (Total)": "#FF4444",
            "Partial Eclipse (Partial)": "#FFCC00",
            "Annular Eclipse (Annular)": "#33B5E5",
            "Hybrid Eclipse (Hybrid)": "#99CC00",
            "Other (Other)": "#666666"
        }
        self.saros_cycle = 6585
        self.target_locations = ["China", "USA", "Japan", "India", "Australia"]

# DataLoader: loads and preprocesses eclipse data
class DataLoader:
    def __init__(self, config):
        self.excel_path = config.excel_path
        self.raw_df = None
        self.processed_df = None
    def load_and_preprocess(self):
        """Load and preprocess raw data: read Excel, clean, and rename columns."""
        try:
            # Check if Excel file exists
            if not os.path.exists(self.excel_path):
                raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

            # Read raw data starting from row 4
            self.raw_df = pd.ExcelFile(self.excel_path).parse(header=None)
            self.processed_df = self.raw_df.iloc[3:].reset_index(drop=True)

            # Assign column names
            self.processed_df.columns = [
                "Calendar Date (Link to Global Map)",
                "TD of Greatest Eclipse",
                "Eclipse Type (Link to Google Map)",
                "Saros Series (Link to Saros)",
                "Eclipse Magnitude",
                "Central Duration (Link to Path Table)",
                "Geographic Region of Eclipse Visibility",
                "Empty Column"
            ]

            # Data cleaning
            date_col = "Calendar Date (Link to Global Map)"
            # Keep rows with at least 4 non-null values
            self.processed_df = self.processed_df.dropna(thresh=4)
            # Filter rows with valid date format (e.g., "2001 Jun 21")
            self.processed_df = self.processed_df[
                self.processed_df[date_col].str.match(r"\d{4} [A-Za-z]{3} \d{1,2}", na=False)
            ]
            # Convert date column to datetime type
            self.processed_df[date_col] = pd.to_datetime(
                self.processed_df[date_col], format="%Y %b %d", errors="coerce"
            )
            # Remove rows where date conversion failed
            self.processed_df = self.processed_df.dropna(subset=[date_col])

            # Rename columns for easier subsequent operations
            self.processed_df = self.processed_df.rename(columns={
                date_col: "Date",
                "Eclipse Type (Link to Google Map)": "Eclipse_Type",
                "Geographic Region of Eclipse Visibility": "Visible_Region"
            })

            print(f"Data preprocessing completed: Number of valid records = {len(self.processed_df)}")
            return self.processed_df
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            print("Please check if the file path is correct, or re-download the data from the link below:")
            print("NASA Solar Eclipse Data Download Link: https://eclipse.gsfc.nasa.gov/SEdecade/SEdecade2021.html")
            return None
        except Exception as e:
            print(f"Data processing failed: {str(e)}")
            return None

# EclipseClassifier: classifies eclipse types and computes statistics
class EclipseClassifier:
    def __init__(self, config, df):
        self.df = df
        self.classification_dict = config.classification_dict
    def classify(self):
        """Classify eclipse types and output category counts."""

        def _classify_single(row):
            """Inner function: Classify a single row of eclipse data."""
            raw_type = row["Eclipse_Type"]
            for category, raw_list in self.classification_dict.items():
                if raw_type in raw_list:
                    return category
            return "Other (Other)"

        self.df["Eclipse_Category"] = self.df.apply(_classify_single, axis=1)
        # Count occurrences of each eclipse type
        category_count = self.df["Eclipse_Category"].value_counts().to_dict()

        # Print statistical results
        print("\nEclipse Type Statistical Results:")
        total = len(self.df)
        for cat, count in category_count.items():
            print(f"  {cat}: {count} times (accounting for {round(count / total * 100, 1)}%)")
        return self.df, category_count

# TipGenerator: generates observation tips via LLM and saves to Excel
class TipGenerator:
    def __init__(self, config):
        self.client = config.client
        self.model = config.model_name
        # Store output directory for saving reports
        self.output_dir = config.output_dir
    def generate(self, category_count):
        """Generate observation tips for all eclipse categories and save to Excel."""
        observation_tips = {}
        print("\nCalling DeepSeek to generate observation tips:")
        for cat in category_count.keys():
            print(f"  Generating observation tips for [{cat}]...")
            try:
                tip = self._generate_observation_tip(cat)
                observation_tips[cat] = tip
                print(f"  Observation tips for [{cat}]:\n{tip}\n")
            except Exception as e:
                observation_tips[cat] = f"Generation failed: {str(e)[:50]}"
                print(f"  Failed to generate observation tips for [{cat}]: {str(e)[:50]}\n")
            time.sleep(1)  # Avoid rate limiting from high-frequency requests

        # Save tips to Excel
        tips_df = pd.DataFrame(
            list(observation_tips.items()),
            columns=["Eclipse Type", "Observation Tips"]
        )
        tips_path = f"{self.output_dir}/eclipse_observation_tips.xlsx"
        tips_df.to_excel(tips_path, index=False, engine="openpyxl")
        print(f"Observation tips saved to: {tips_path}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=False
    )
    def _generate_observation_tip(self, eclipse_category):
        """Private method: Call LLM to generate observation tips for a single eclipse type (with retry mechanism)."""
        prompt = f"""As an astronomy popularization expert, provide observation tips for 【{eclipse_category}】, strictly following:
1. 3 points: Safety Protection, Equipment Recommendations, Timing Selection;
2. Each point no more than 40 characters, concise and practical language;
3. Combine with observation scenarios of this eclipse type from 2001 to 2025.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Only retain observation tips, no extra expansion, colloquial language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

# Visualizer: generates frequency and regional pie charts
class Visualizer:
    def __init__(self, config, df, category_count, target_locations, color_map, start_year, end_year):
        self.df = df
        self.category_count = category_count
        self.target_locations = target_locations
        self.color_map = color_map
        self.output_dir = config.output_dir
        # Analysis period
        self.start_year = start_year
        self.end_year = end_year
        # Store analysis period
        self.start_year = start_year
        self.end_year = end_year
    def plot_frequency(self):
        """Plot time-frequency and type distribution charts."""
        # Add year column for time-based grouping
        self.df["Year"] = self.df["Date"].dt.year
        # Count frequency by year and eclipse type
        yearly_freq = self.df.groupby(["Year", "Eclipse_Category"]).size().unstack(fill_value=0)

        # Configure font to support English (no need for Chinese font anymore)
        plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Subplot 1: Time-frequency stacked bar chart
        colors = [self.color_map.get(col, "#666") for col in yearly_freq.columns]
        yearly_freq.plot(
            kind="bar", stacked=True, ax=ax1, color=colors,
            title=f"Eclipse Frequency Trend ({self.start_year}-{self.end_year})",
            xlabel="Year", ylabel="Number of Occurrences"
        )
        ax1.legend(title="Eclipse Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(axis="y", alpha=0.3)
        ax1.title.set_fontsize(16)
        ax1.title.set_fontweight("bold")

        # Subplot 2: Eclipse type distribution bar chart
        cats = list(self.category_count.keys())
        counts = [self.category_count[cat] for cat in cats]
        bars = ax2.bar(cats, counts, color=[self.color_map.get(cat, "#666") for cat in cats])
        # Add numerical labels to bars
        for bar, count in zip(bars, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold"
            )
        ax2.set_title(f"Eclipse Type Distribution ({self.start_year}-{self.end_year})", fontsize=16, fontweight="bold")
        ax2.set_xlabel("Eclipse Type")
        ax2.set_ylabel("Number of Occurrences")
        ax2.grid(axis="y", alpha=0.3)

        # Save plot
        plt.tight_layout()
        chart_path = f"{self.output_dir}/eclipse_frequency_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nTime-frequency chart saved to: {chart_path}")

    def plot_pie(self):
        """Plot regional eclipse type distribution pie charts."""
        # CN-EN region mapping (for keyword matching and filename)
        cn_en_map = {
            "China": ["China", "Chinese"],
            "USA": ["USA", "United States", "America"],
            "Japan": ["Japan", "Japanese"],
            "India": ["India", "Indian"],
            "Australia": ["Australia", "Australian"]
        }

        print("\nBuilding regional solar eclipse records:")
        for location in self.target_locations:
            # Get keywords for region matching
            keywords = cn_en_map.get(location, [location])
            # Fuzzy match visible regions
            match_mask = pd.Series(False, index=self.df.index)
            for kw in keywords:
                match_mask |= self.df["Visible_Region"].str.contains(kw, case=False, na=False)

            # Filter and sort data for the current region
            loc_df = self.df[match_mask].sort_values("Date").reset_index(drop=True)
            self.location_records[location] = loc_df

            if len(loc_df) > 0:
                # Save regional records to Excel
                loc_en = cn_en_map.get(location, [location])[0]
                loc_path = f"{self.output_dir}/eclipses_in_{loc_en}.xlsx"
                loc_df.to_excel(loc_path, index=False, engine="openpyxl")

                # Plot pie chart for eclipse type distribution
                loc_cat_count = loc_df["Eclipse_Category"].value_counts()
                pie_colors = [self.color_map.get(cat, "#666") for cat in loc_cat_count.index]

                plt.figure(figsize=(10, 7))
                wedges, texts, autotexts = plt.pie(
                    loc_cat_count.values, labels=loc_cat_count.index,
                    autopct="%1.1f%%", colors=pie_colors, startangle=90
                )
                # Optimize text style
                plt.setp(texts, fontsize=11)
                plt.setp(autotexts, fontsize=10, fontweight="bold", color="white")
                plt.title(
                    f"Eclipse Type Distribution in {location} ({self.start_year}-{self.end_year})",
                    fontsize=14, fontweight="bold"
                )

                # Save pie chart
                pie_path = f"{self.output_dir}/{loc_en}_eclipse_distribution.png"
                plt.savefig(pie_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Print success message
                print(f"  {location}: {len(loc_df)} matching records found")
                print(f"     - Matching keywords: {', '.join(keywords)}")
                print(f"     - Record file: {loc_path}")
                print(f"     - Pie chart file: {pie_path}")
            else:
                print(f"  {location}: No matching eclipse records found")
                print(f"     - Attempted matching keywords: {', '.join(keywords)}")
                print(f"     - Suggestion: Check if these keywords exist in the 'Visible_Region' column")

# CycleDetector: detects potential eclipse cycles and Saros matches
class CycleDetector:
    def __init__(self, config, df):
        self.df = df
        self.saros_cycle = config.saros_cycle
        self.output_dir = config.output_dir
    def detect(self):
        """Detect potential repeating intervals and Saros cycle matches."""
        # The Saros cycle is an important astronomical cycle, approximately 6585.32 days (18 years, 11 days, 8 hours)
        # Eclipses in the same Saros cycle have similar geometric characteristics and reappear in different regions of Earth.

        # 1. Sort data by date and calculate intervals between consecutive eclipses
        # Sorting ensures intervals are calculated for chronologically consecutive eclipses
        sorted_df = self.df.sort_values("Date").reset_index(drop=True)
        # Calculate time difference (in days) from the previous eclipse
        sorted_df["Interval_Days"] = (sorted_df["Date"] - sorted_df["Date"].shift(1)).dt.days

        # 2. Identify potential cycle patterns
        # Filter intervals that occur at least twice (exclude 0 days to avoid duplicate records on the same day)
        interval_counts = sorted_df["Interval_Days"].value_counts()
        potential_cycles = interval_counts[interval_counts >= 2].to_dict()

        # 3. Verify Saros cycle matches
        # Allow ±10 days tolerance (account for minor deviations in actual astronomical observations)
        tolerance = 10  # Tolerance for Saros cycle matching (in days)
        saros_matches = sorted_df[
            (sorted_df["Interval_Days"] >= self.saros_cycle - tolerance) &
            (sorted_df["Interval_Days"] <= self.saros_cycle + tolerance)
            ]["Interval_Days"].tolist()

        # 4. Print detection results
        print("\nEclipse Cycle Detection Results:")
        print(f"  Potential repeating intervals (days) and occurrence count:")
        # Sort by occurrence count in descending order
        for interval, count in sorted(potential_cycles.items(), key=lambda x: x[1], reverse=True):
            print(f"    {interval} days: {count} times")
        print(f"  Saros cycle (theoretical value: {self.saros_cycle} days) matching results:")
        print(f"    Qualified intervals: {saros_matches}")
        print(f"    Matching count: {len(saros_matches)} times (tolerance: ±{tolerance} days)")

        # Save cycle detection results to Excel
        try:
            cycle_df = pd.DataFrame(list(potential_cycles.items()), columns=['Interval_Days', 'Count'])
            matches_df = pd.DataFrame({'Matched_Saros_Interval': saros_matches})
            cycle_path = os.path.join(self.output_dir, 'eclipse_cycle_detection.xlsx')
            with pd.ExcelWriter(cycle_path, engine='openpyxl') as writer:
                cycle_df.to_excel(writer, sheet_name='Interval_Counts', index=False)
                matches_df.to_excel(writer, sheet_name='Saros_Matches', index=False)
            print(f"Cycle detection results saved to: {cycle_path}")
        except Exception as e:
            print(f"Failed to save cycle detection results: {e}")
        return potential_cycles, saros_matches

# EventMatcher: matches eclipses with historical events
class EventMatcher:
    def __init__(self, config, df):
        self.df = df
        self.historical_events = config.historical_events
    def match(self):
        """Match eclipse dates to historical events."""
        # Convert date format to "YYYY-MM-DD" for easy matching
        self.df["Date_Str"] = self.df["Date"].dt.strftime("%Y-%m-%d")

        print("\nHistorical Event Matching Results:")
        event_matches = []
        for event_date, event_desc in self.historical_events.items():
            # Exact date matching
            match_mask = self.df["Date_Str"] == event_date
            if match_mask.any():
                # Get detailed information of the matching eclipse
                eclipse_info = self.df[match_mask].iloc[0]
                event_matches.append({
                    "Eclipse Date": event_date,
                    "Eclipse Type": eclipse_info["Eclipse_Category"],
                    "Visible Region": eclipse_info["Visible_Region"],
                    "Associated Historical Event": event_desc
                })
                print(f"  Match successful: {event_date} - {event_desc}")
            else:
                print(f"  Match failed: {event_date} - {event_desc} (no eclipse record on this date)")

        return event_matches

# CulturalAnalyzer: compares cultural interpretations via LLM
class CulturalAnalyzer:
    def __init__(self, config, category_count, start_year, end_year):
        self.client = config.client
        self.model = config.model_name
        self.category_count = category_count
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = config.output_dir
        # Store analysis period and output directory
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = config.output_dir
    def analyze(self):
        """Generate cross-cultural eclipse interpretations and save to TXT."""
        # Select 4 representative cultures
        cultures = ["Ancient China", "Ancient Egypt", "Maya Civilization", "Medieval Europe"]
        eclipse_categories = list(self.category_count.keys())

        print("\nGenerating cross-cultural eclipse interpretation comparison:")
        cultural_analyses = {}
        for culture in cultures:
            print(f"  Generating interpretation for [{culture}]...")
            try:
                analysis = self._generate_cultural_analysis(culture, eclipse_categories)
                cultural_analyses[culture] = analysis
                print(f"  Interpretation for [{culture}] completed")
            except Exception as e:
                cultural_analyses[culture] = f"Generation failed: {str(e)[:50]}"
                print(f"  Failed to generate interpretation for [{culture}]: {str(e)[:50]}")
            time.sleep(1.5)  # Add delay to avoid rate limiting

        # Save interpretation results to TXT
        analysis_path = f"{self.output_dir}/eclipse_cultural_comparison.txt"
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(f"Cross-Cultural Eclipse Interpretations ({self.start_year}-{self.end_year})\n")
            f.write("=" * 50 + "\n\n")
            for culture, content in cultural_analyses.items():
                f.write(f"[{culture}]\n")
                f.write(content + "\n\n")
                f.write("-" * 30 + "\n\n")

        print(f"\nCultural interpretation comparison saved to: {analysis_path}")
        return cultural_analyses

# DataSaver: saves fully processed data to Excel
class DataSaver:
    def __init__(self, config, df):
        self.df = df
        self.output_dir = config.output_dir
    def save_full(self):
        """Save the complete processed eclipse dataset."""
        full_path = f"{self.output_dir}/full_processed_eclipse_data.xlsx"
        self.df.to_excel(full_path, index=False, engine="openpyxl")
        print(f"\nAll analyses completed! Fully processed data saved to: {full_path}")
        print(f"All result files are located at: {os.path.abspath(self.output_dir)}")

# Orchestrator: orchestrates the full workflow by invoking all components
class Orchestrator:
    def __init__(self):
        self.config = ConfigManager()
    def run(self):
        # 1. Load and preprocess data
        df = DataLoader(self.config).load_and_preprocess()

        # 2. Time range selection
        print("\nPlease select the time range for analysis:")
        print("1. 2001 - 2025 (All data)")
        print("2. Custom time range")
        choice = input("Please select an option (1 or 2): ").strip()
        if choice == "1":
            start_year, end_year = 2001, 2025
        elif choice == "2":
            try:
                start_year = int(input("Please enter the start year (2001-2025): ").strip())
                end_year = int(input("Please enter the end year (2001-2025): ").strip())
                if not (2001 <= start_year <= 2025 and 2001 <= end_year <= 2025 and start_year <= end_year):
                    print("Invalid range, using default (2001-2025)")
                    start_year, end_year = 2001, 2025
            except ValueError:
                print("Invalid input, using default (2001-2025)")
                start_year, end_year = 2001, 2025
        else:
            print("Invalid input, using default (2001-2025)")
            start_year, end_year = 2001, 2025

        # Filter data by selected range
        df = df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()
        print(f"\nAnalysis period: {start_year}-{end_year}, records: {len(df)}")

        # 3. Classify eclipses
        df, category_count = EclipseClassifier(self.config, df).classify()

        # 4. Generate observation tips
        TipGenerator(self.config).generate(category_count)

        # 5. Visualization
        vis = Visualizer(
            self.config,
            df,
            category_count,
            self.config.target_locations,
            self.config.color_map,
            start_year,
            end_year
        )
        vis.plot_frequency()
        vis.plot_pie()

        # 6. Cycle detection and historical event matching
        CycleDetector(self.config, df).detect()
        EventMatcher(self.config, df).match()

        # 7. Cross-cultural interpretation
        CulturalAnalyzer(self.config, category_count, start_year, end_year).analyze()

        # 8. Save final data
        DataSaver(self.config, df).save_full()

# -------------------------- Execute Analysis --------------------------
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = Orchestrator()
    # Run full workflow analysis
    analyzer.run()