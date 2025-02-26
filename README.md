This is a markdown file providing a description of the project, how to run the code, and any relevant information for someone to understand and use the repository. 
# Fake-News-Project
This repository contains code and analysis for a text classification project, focusing on identifying fake news articles using machine learning techniques. The project explores data preparation, Zipf's Law analysis, text vectorization, and compares the performance of Naive Bayes, SVM, and Decision Tree classification models.

# Text Classification: Fake News Detection

This project aims to classify news articles as fake or real using machine learning.

## Project Overview

*   **Dataset:** Fake and Real News Dataset from Kaggle
*   **Techniques:**
    *   Data cleaning and preparation
    *   Zipf's Law analysis
    *   Text vectorization (TF-IDF)
    *   Naive Bayes classification
    *   SVM classification
    *   Decision Tree classification
*   **Evaluation:** Accuracy, confusion matrix, precision, recall, F1-score

## Files

*   `Fake_News_Project.ipynb`: Jupyter Notebook containing the code and analysis.
*   `Fake.zip`: Dataset file for "fake" news. 
*   `Real.zip`: Dataset file for "real" news. 

## Running the Code

1.  Install the required libraries: `scikit-learn`, `pandas`, `matplotlib`.
2.  Download the dataset and place it in the same directory as the notebook.
3.  Open and run the Jupyter Notebook ('Fake_News_Project.ipynb') to see the analysis and results.

## Results

*   The SVM model achieved the highest accuracy of 99.52%.
*   Further analysis and model comparisons are detailed in the notebook.

## Future Work

*   Explore ensemble models (Random Forest, Gradient Boosting).
*   Fine-tune hyperparameters for improved performance.
*   Investigate additional features for enhanced classification.

## References

* Bisaillon, C. (2020). Fake and real news dataset. Kaggle. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to information retrieval. Cambridge University Press.
* Zipf, G. K. (1949). Human behavior and the principle of least effort. Addison-Wesley Press. 
