# Churn Predictor using FakeStoreAPI Data

## Project Overview

This project implements a Customer Churn Prediction System using data inspired by the structure of FakeStoreAPI. The objective is to identify customers who are likely to stop interacting with an e-commerce platform based on their purchase history, cart activity, and account information.

The solution follows a modular architecture that separates:
- Data collection and simulation
- Feature engineering
- Feature selection and exploratory analysis
- Machine learning model training
- REST API deployment with FastAPI
- Containerization using Docker

---

## Project Structure

```text
churn-predictor/
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # ML training and prediction logic
│   ├── features.py      # Feature engineering pipeline
|   ├── __init__.py      # Constructor for the class
│   └── scraper.py       # Data collection and simulation
│
├── notebooks/
│   └── eda_and_selection.ipynb
│
├── data/
│   └── raw/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── pyproject.toml
```

---

## System Architecture

### scraper.py

Responsible for collecting or simulating raw data based on the FakeStoreAPI schema.

Functions:

- Generate user profiles
- Simulate shopping carts
- Simulate product purchases
- Generate churn labels
- Save datasets in `data/raw/`

This module contains **no machine learning logic**.

---

### features.py

Responsible for transforming raw e-commerce data into machine learning features.

---

### model.py

Responsible for:

- Loading engineered features
- Training the machine learning model
- Evaluating model performance
- Saving trained models using Joblib
- Loading models for inference

---

### main.py

FastAPI application layer.

Responsibilities:

- Request validation using Pydantic
- Feature generation from incoming raw customer data
- Calling the prediction pipeline
- Returning churn probability and predicted class

---

## Churn Definition

A customer is considered churned when:

> No cart activity has been detected during the last 6 months.

Where:

- `0` = Active customer
- `1` = Churned customer

---

## Dataset

The dataset is generated from three entities inspired by FakeStoreAPI.

### Users

```json
{
  "id": 1,
  "email": "user@mail.com",
  "registered_at": "2024-01-01",
  "address": {
    "city": "New York",
    "zipcode": "10001"
  }
}
```

### Carts

```json
{
  "id": 1,
  "userId": 1,
  "date": "2025-01-10",
  "products": [...]
}
```

### Products

```json
{
  "productId": 15,
  "category": "electronics",
  "price": 399.99,
  "quantity": 2
}
```

---

## Feature Engineering

The model uses several feature types.

### Time Features

- `recency_days`
- `account_age_days`

### Aggregation Features

- `total_spent`
- `cart_count`
- `avg_items_per_cart`
- `distinct_categories`

### Ratio Features

- `expensive_ratio`

### Binary Features

- `bought_electronics`
- `has_address`

---

## Feature Selection

The notebook evaluates features using four different selection techniques.

 1. Filter Method

 2. Wrapper Method

 3. Decision Tree Feature Importance

 4. Random Forest Feature Importance

A consolidated comparison table is generated to identify the most relevant features across all methods.

---

## Installation

### Clone the Repository

```bash
git clone <repository_url>
cd churn-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Generate Dataset

```bash
python app/scraper.py
```

Generated files:

```text
data/raw/users.json
data/raw/carts.json
data/raw/labels.csv
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

**Request**

```http
GET /health
```

**Response**

```json
{
  "status": "ok"
}
```

---

### Predict Churn

**Request**

```http
POST /predict
```

Example payload:

```json
{
  "user": {
    "id": 1,
    "email": "user1@mail.com",
    "registered_at": "2024-01-01T00:00:00",
    "address": {
      "city": "New York",
      "zipcode": "10001"
    }
  },
  "carts": [
    {
      "id": 1,
      "userId": 1,
      "date": "2025-01-10T00:00:00",
      "products": [
        {
          "productId": 10,
          "category": "electronics",
          "price": 299.99,
          "quantity": 1
        }
      ]
    }
  ]
}
```

Example response:

```json
{
  "churn_probability": 0.18,
  "churn_class": 0
}
```

---

## Docker Deployment

### Build Image

```bash
docker compose build
```

### Start Container

```bash
docker compose up
```

The application will be available at:

```text
http://localhost:8000
```

---

## Technologies Used

- Python 3.10
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Uvicorn
- Docker
- Jupyter Notebook
- Matplotlib

---


Customer Churn Prediction Project

Software Engineering and Data Science project demonstrating the complete lifecycle of a machine learning solution, from data generation and feature engineering to model deployment using FastAPI and Docker.