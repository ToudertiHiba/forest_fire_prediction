# Forest Fire Prediction Project

This project focuses on predicting forest fires using machine learning techniques, specifically a neural network regression model. The dataset used for this project is sourced from Kaggle.

## Overview

The goal of this project is to develop a predictive model that can estimate the potential area of a forest fire based on various environmental features. The project uses a neural network regression model implemented with TensorFlow/Keras.

## Files and Directories

- `forest_fire_prediction.ipynb`: Jupyter Notebook containing the Python code for the project.
- `forest_fire_prediction.py`: A python file that contains a slightly varied version of the Jupyter Notebook.
- `forestfires.csv`: The dataset used for training and testing the model, obtained from Kaggle.
- `pyproject.toml`: Poetry configuration file specifying project dependencies.
- `README.md`: This file, providing an overview of the project.

## Getting Started

To run this project, follow these steps:

1. **Set Up the Environment**:

   - Make sure you have Python installed on your system.
   - Install Poetry, a package manager for Python, using the instructions provided in the [official documentation](https://python-poetry.org/docs/).

2. **Install Dependencies**:

   Navigate to the project directory and run:

   ```bash
   poetry install
   ```

3. **Data Preparation**:

   Place the dataset `forestfires.csv` in the project directory.

4. **Running the Jupyter Notebook**:

   Open and run the `forest_fire_prediction.ipynb` notebook using Jupyter or any compatible environment.

## Project Structure

- The Jupyter Notebook `forest_fire_prediction.ipynb` contains the entire codebase organized into sections with comments.
- The dataset `forestfires.csv` contains the necessary data for training and testing the model.

## Dependencies

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `sklearn`
- `tensorflow`

All dependencies are managed via [Poetry](https://python-poetry.org/). You can find the complete list of dependencies in the `pyproject.toml` file.

## Notes

- This project is a basic implementation of a forest fire prediction model. Further improvements can be made by experimenting with different algorithms, feature engineering, and hyperparameter tuning.
