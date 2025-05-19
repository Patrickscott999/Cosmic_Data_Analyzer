# 🌌 Cosmic Data Analyzer
https://cosmicdataanalyzer.replit.app/


A sophisticated AI-powered data analysis platform that transforms complex datasets into actionable insights through advanced machine learning techniques, visualization, and intelligent recommendations.

![Cosmic Data Analyzer](https://i.imgur.com/your-project-screenshot.png)

## ✨ Features

### Data Processing
- **Advanced Data Cleaning**: Automatic handling of missing values, outlier detection, data normalization, and type conversion
- **Smart Data Preparation**: Automated feature selection and encoding for machine learning workflows
- **Data Export**: Export your processed data in multiple formats (CSV, Excel, JSON)

### Analytics & Machine Learning
- **Automated ML Models**: Train classification and regression models with automatic hyperparameter tuning
- **Data Mining**: Discover patterns with Apriori association rules and KMeans clustering
- **Time Series Forecasting**: Predict future trends in your temporal data

### Visualization
- **Interactive Charts**: Generate beautiful visualizations to identify patterns
- **Correlation Analysis**: Visualize relationships between variables with heatmaps
- **Custom Plot Styling**: Adjust colors, labels, and annotations for your visualizations

### AI-Powered Insights
- **Natural Language Queries**: Ask questions about your data in plain English
- **Automated Insights**: Get AI-generated observations about patterns in your data
- **Anomaly Detection**: Automatically identify outliers and unusual patterns
- **Data Quality Assessment**: Get recommendations for improving your dataset

## 🚀 Technologies Used

- **Backend**: Python, Flask, SQLAlchemy, PostgreSQL
- **Data Science**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning**: Classification, Regression, Clustering, Association Rules
- **AI Integration**: OpenAI GPT-4o for natural language understanding and insights
- **Authentication**: Secure user authentication via Replit Auth
- **UI**: HTML5, CSS3, JavaScript with Bootstrap 5 and custom animations

## 🔮 Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL database
- OpenAI API key (for AI features)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cosmic-data-analyzer.git
   cd cosmic-data-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   DATABASE_URL=your_postgresql_database_url
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Initialize the database:
   ```bash
   flask db upgrade
   ```

5. Run the application:
   ```bash
   python main.py
   ```

6. Visit `http://localhost:5000` in your browser to see the application.

## 💫 Usage Guide

### Upload Your Data
1. Log in to your account
2. Click "Upload New Dataset" on the dashboard
3. Upload your CSV, Excel, or JSON file
4. Your dataset will be automatically analyzed and previewed

### Clean Your Data
1. Select your dataset
2. Click "Clean Data" tab
3. Choose the cleaning options (handle nulls, normalize, etc.)
4. Apply the changes

### Train Machine Learning Models
1. Navigate to the "ML Models" tab
2. Select your target variable
3. Choose the model type (classification/regression)
4. Train and evaluate your model

### Generate Visualizations
1. Go to the "Visualize" tab
2. Select the chart type and variables
3. Customize your visualization
4. Export as image or interactive HTML

### Get AI Insights
1. Select the "AI Insights" tab
2. Choose the type of analysis
3. View the AI-generated insights and recommendations

## 🌠 Roadmap

- Integration with more data sources (APIs, databases)
- Real-time collaboration features
- Advanced neural network models
- Custom report generation
- Mobile application

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgements

- [OpenAI](https://openai.com/) for GPT-4o API
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Pandas](https://pandas.pydata.org/) for data processing
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualizations

---

<p align="center">Explore the Universe of Data with Cosmic Data Analyzer ✨🔭</p>
