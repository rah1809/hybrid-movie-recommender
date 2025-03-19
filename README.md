# 🎥 Hybrid Movie Recommender Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)](https://streamlit.io/)
[![GitHub stars](https://img.shields.io/github/stars/rah1809/hybrid-movie-recommender)](https://github.com/rah1809/hybrid-movie-recommender/stargazers)

---








# 🎥 Hybrid Movie Recommender Engine

A sophisticated movie recommendation system that combines multiple recommendation algorithms to provide personalized movie suggestions. Built with Streamlit, this interactive application allows users to explore different recommendation approaches and evaluate their performance.

## ✨ Key Features

- **Multiple Recommendation Algorithms:**
  - 🎯 Trivial Recommender (Popular movies)
  - 📈 Most Popular Items (MPI)
  - 👥 User-based Collaborative Filtering
  - 🎬 Item-based Collaborative Filtering
  - 🔢 SVD (Matrix Factorization)
- **Interactive UI:**
  - Easy-to-use sidebar controls
  - Real-time recommendations
  - Expandable movie cards with ratings and genres
- **Evaluation Dashboard:**
  - RMSE & MAE comparisons
  - Precision@K & Recall@K metrics
  - Interactive visualizations using Plotly
  - Detailed performance analysis

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Interactive web interface
- **Surprise** - Recommendation algorithms
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computations

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hybrid-movie-recommender.git
cd hybrid-movie-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

## 🚀 Usage

### Settings Panel
- Choose from 5 different recommendation algorithms
- Enter User ID (1-6040)
- Select number of recommendations (1-20)
- Click "Get Recommendations" to generate personalized suggestions

### Instructions
1. Select a recommendation algorithm from the dropdown
2. Enter a User ID to get personalized recommendations
3. Adjust the number of recommendations using the slider
4. Click "Get Recommendations" to see results
5. Switch to "Evaluation Dashboard" to compare algorithm performance

### Evaluation Dashboard
- Compare algorithm performance using various metrics
- View interactive bar charts for RMSE and MAE
- Analyze Precision@K and Recall@K scores
- Explore detailed metrics table

## 📊 Model Performance

The system evaluates recommender performance using:
- **RMSE (Root Mean Square Error)** - Prediction accuracy
- **MAE (Mean Absolute Error)** - Average prediction deviation
- **Precision@5** - Relevance of recommendations
- **Recall@5** - Coverage of relevant items

## 👤 Credits

Developed by Rahmat Khan

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Rahmat Khan

Permission is not granted to any person obtaining a copy of this software and associated documentation files (the "Software") to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, or to permit persons to whom the Software is furnished to do so.

The Software is not permitted for any use, modification, or redistribution.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


