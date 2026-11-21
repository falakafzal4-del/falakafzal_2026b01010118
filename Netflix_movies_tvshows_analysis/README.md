# Netflix Movies & TV Shows Analysis

## Project Overview

This project focuses on Exploratory Data Analysis (EDA) of Netflix Movies and TV Shows using Python. The objective is to analyze Netflix's content catalogue, identify trends, and discover meaningful insights through data cleaning, data analysis, and visualization.

The dataset contains information about titles, directors, cast, countries, release years, ratings, durations, and genres.

## Project Objectives

- Analyze the distribution of Movies and TV Shows.
- Identify the countries contributing the most titles.
- Explore popular content genres.
- Understand content rating distribution.
- Analyze movie durations and TV show seasons.
- Study content addition trends over time.
- Create visualizations to present meaningful insights.

## Dataset Overview

- **Total Records:** 8,807
- **Total Columns:** 12

### Dataset Columns

| Column | Description |
|---|---|
| show_id | Unique ID of each title |
| type | Movie or TV Show |
| title | Title name |
| director | Director name |
| cast | Cast members |
| country | Country of production |
| date_added | Date added to Netflix |
| release_year | Original release year |
| rating | Content rating |
| duration | Movie duration or TV show seasons |
| listed_in | Genre or category |
| description | Short description of the title |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Anaconda

## Project Structure

```text
Netflix_Movies_TV_Shows_Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── netflix_analysis.ipynb
│
├── images/
│   └── netflix_analysis_dashboard.png
│
├── README.md
└── requirements.txt
```

## Project Workflow

1. **Data Loading:** Imported the dataset using Pandas.
2. **Data Inspection:** Examined dataset structure, columns, data types, missing values, and duplicates.
3. **Data Cleaning:** Handled missing values and corrected data inconsistencies.
4. **Exploratory Data Analysis:** Analyzed content types, release years, countries, genres, ratings, movie durations, TV show seasons, and content addition trends.
5. **Data Visualization:** Created charts and a combined dashboard using Matplotlib and Seaborn.
6. **Insights:** Identified and summarized important patterns in the dataset.

## Key Findings

- Total titles analyzed: **8,807**
- Movies: **6,131 (69.62%)**
- TV Shows: **2,676 (30.38%)**
- Movies are approximately 2.29 times as numerous as TV Shows.

## Dashboard

The project includes a dashboard featuring four visualizations:

- Movies vs TV Shows
- Top 10 Countries
- Top 10 Genres
- Top 10 Content Ratings

![Netflix Analysis Dashboard](images/netflix_analysis_dashboard.png)

## How to Run This Project

1. Clone this repository:

   ```bash
   git clone https://github.com/divyanshshankhdhar/Netflix-Movies-TV-Shows-Analysis.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Netflix-Movies-TV-Shows-Analysis
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset file `netflix_titles.csv` inside the `data/raw/` folder.

5. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `notebooks/netflix_analysis.ipynb` and execute the cells in order.

## Future Improvements

- Develop an interactive dashboard using Power BI or Streamlit.
- Perform deeper analysis of content trends.
- Explore relationships between release years, genres, and ratings.
- Add more visualizations and business insights.
- Explore machine learning applications where appropriate.

## Disclaimer

This project is created for educational and portfolio purposes. The analysis is based on the available dataset and does not represent Netflix's official internal business data.

## Author

**Falak Afzal**

B.Tech Computer Science Engineering Student

GitHub: GitHub: [falakafzal4-del](https://github.com/falakafzal4-del)
