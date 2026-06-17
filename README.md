# 📚 PageTurn - Hybrid Book Recommendation System

## Overview

PageTurn is a Hybrid Collaborative Filtering Book Recommendation System built using Python, Pandas, Scikit-Learn, and Streamlit.

The system analyzes user reading behavior and book ratings to recommend books that users are likely to enjoy.

Unlike traditional recommendation systems that rely on a single method, PageTurn combines:

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Hybrid Recommendation Scoring

This approach improves recommendation quality and personalization.

---

## Features

### Data Processing
- User dataset cleaning
- Age validation and preprocessing
- Location extraction (City, State, Country)
- Missing value handling
- Rating filtering
- Active user filtering
- Popular book filtering

### Exploratory Data Analysis (EDA)
- User age group analysis
- Book popularity analysis
- Rating distribution analysis
- Reader behavior analysis
- Top-rated and most-read books visualization

### Recommendation Engine

#### User-Based Collaborative Filtering
Finds users with similar reading preferences and recommends books they enjoyed.

#### Item-Based Collaborative Filtering
Finds books similar to books already liked by the target user.

#### Hybrid Recommendation Model
Combines both recommendation strategies using weighted scores.

Final Score:

Final Score =
(Item Similarity × Item Weight)
+
(User Similarity × User Weight)

Default Weights:

- Item-Based = 70%
- User-Based = 30%

---

## Dataset

The project uses the Book-Crossing Dataset.

Files:

- Users.csv
- Books.csv
- Ratings.csv

### Dataset Components

#### Users
Contains:
- User-ID
- Location
- Age

#### Books
Contains:
- ISBN
- Book Title
- Author
- Publisher
- Image URLs

#### Ratings
Contains:
- User-ID
- ISBN
- Book Rating

---

## Data Cleaning Steps

### Users Dataset

- Converted Age to numeric format
- Removed unrealistic age values
- Created Age Groups
- Extracted:
  - City
  - State
  - Country

### Ratings Dataset

Removed:
- Ratings equal to 0
- Inactive users

Kept:
- Users with at least 20 ratings

### Books Dataset

Handled:
- Missing authors
- Missing publishers
- Invalid publication years

### Popularity Filtering

Kept books having:

- At least 5 ratings (Web App)
- At least 10 ratings (EDA Notebook)

This reduces noise and improves recommendation quality.

---

## Machine Learning Approach

### Pivot Matrix Creation

User-Book Matrix

Rows:
- Users

Columns:
- Books

Values:
- Ratings

Missing values replaced with:

0

---

### Similarity Computation

Cosine Similarity was used for:

#### User Similarity Matrix

Measures similarity between users.

#### Book Similarity Matrix

Measures similarity between books.

---

## Recommendation Workflow

### Step 1

User enters User ID.

### Step 2

System retrieves user's reading history.

### Step 3

Generate:

- User-Based Recommendations
- Item-Based Recommendations

### Step 4

Combine scores using weighted hybrid ranking.

### Step 5

Return Top-N personalized books.

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Matplotlib
- Seaborn
- Streamlit

### Recommendation Techniques

- Collaborative Filtering
- Cosine Similarity
- Hybrid Recommendation Systems

---

## Streamlit Application Features

### Reader Dashboard

- Enter User ID
- Select recommendation count
- Adjust recommendation weights

### User Storyboard

Displays:

- Active readers
- Books read
- Average ratings
- Country information

### BookGalaxy Showcase

Displays:

- Popular books
- Book covers
- Average ratings
- Reader counts

### Personalized Recommendations

Shows:

- Book cover
- Book title
- Author
- Publisher
- Match score

---

## Project Structure

├── app.py
├── Book Recommendation System.ipynb
├── Users.csv
├── Books.csv
├── Ratings.csv
├── README.md

---

## Dataset Downloads

Users - https://drive.google.com/file/d/174PYOjDf-XNbwGg1XwXB3qpkVhfYOo0v/view?usp=drive_link
Ratings - https://drive.google.com/file/d/1YQp1d3m9nfN-xEeQyMJaJCv0isKJtmUQ/view?usp=drive_link
Books - https://drive.google.com/file/d/1CF4J9XLPxE7jF4jBFpWf7Y3Wy46i--OU/view?usp=drive_link

## Future Improvements

- Content-Based Filtering
- Deep Learning Recommendation Models
- NLP-based Book Description Analysis
- Real-time Recommendation API
- User Authentication
- Cloud Deployment
- Recommendation Evaluation Metrics

---

## Results

The system successfully:

✔ Cleans and processes large-scale book data

✔ Builds user and item similarity models

✔ Generates personalized recommendations

✔ Combines multiple recommendation approaches

✔ Provides an interactive recommendation dashboard

---

## Author

Dongri Ashmanjum

B.Sc Computer Science

Aspiring Data Engineer / AI Engineer

GitHub Portfolio Project
