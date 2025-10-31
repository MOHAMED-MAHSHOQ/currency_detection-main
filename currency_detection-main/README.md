# What I did — Currency Detection (exact steps implemented)

This file lists only the work implemented in the notebook and code — kept short and exact.

- Installed required packages used in the notebook (pandas, scikit-learn, matplotlib, seaborn, opencv-python-headless, scipy).
- Loaded `BankNote_Authentication.csv` and inspected the data (`head()`, `columns`, cleaned column names).
- Prepared features and label: `X = df.drop('class', axis=1)`, `y = df['class']`; printed data shape.
- Split data into training and test sets with `train_test_split(test_size=0.2, random_state=42)`.
- Trained a `RandomForestClassifier(n_estimators=100, random_state=42)` on the training data.
- Predicted on the test set and printed accuracy and classification report.
- Plotted a confusion matrix using seaborn heatmap.
- Saved the trained model to `currency_auth_model.pkl` using `joblib` and demonstrated loading it back.
- Implemented `extract_image_features(image)` to compute variance, skewness, kurtosis and entropy from an input image.
- Added code to upload an image, convert it to grayscale with OpenCV, extract the four features, reshape them, and predict (Real / Fake) using the saved model.

Files referenced/produced in the project:

- `currency_detection.ipynb` — main notebook containing the code above.
- `BankNote_Authentication.csv` — dataset used.
- `currency_auth_model.pkl` — model saved via joblib (created by the notebook).

How to reproduce (very short): open `currency_detection.ipynb` and run the cells top-to-bottom; the notebook contains the exact commands used.
