# Hybrid Movie Recommendation Engine

A sophisticated movie recommendation system combining collaborative filtering, content-based filtering, and matrix factorization approaches using the MovieLens dataset.

## Project Structure
```
📁 recommender-system-project
├── data/                      # Data directory
│   └── ml-1m.zip             # MovieLens 1M dataset
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_DataExploration.ipynb
│   └── 02_Trivial_MPI_CF_SVD.ipynb
├── src/                      # Source code
│   ├── recommenders.py       # Recommendation algorithms
│   └── evaluation.py         # Evaluation metrics
├── streamlit_app/           # Interactive web application
│   └── app.py               # Streamlit application
├── report/                  # Project documentation
├── presentation/           # Project presentation materials
└── requirements.txt        # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract the MovieLens dataset:
```bash
cd data
unzip ml-1m.zip
```

## Components

### 1. Data Exploration (notebooks/01_DataExploration.ipynb)
- Dataset analysis
- Data preprocessing
- Feature engineering
- Visualization of user-movie interactions

### 2. Recommendation Algorithms (notebooks/02_Trivial_MPI_CF_SVD.ipynb)
- Collaborative Filtering (CF)
- Matrix Factorization
- Singular Value Decomposition (SVD)
- Performance optimization using MPI

### 3. Evaluation Metrics (src/evaluation.py)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)

### 4. Interactive Demo (streamlit_app/app.py)
- Web interface for real-time recommendations
- User preference input
- Visualization of recommendation results

## Usage

1. Data Exploration:
```bash
jupyter notebook notebooks/01_DataExploration.ipynb
```

2. Run Streamlit App:
```bash
cd streamlit_app
streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 