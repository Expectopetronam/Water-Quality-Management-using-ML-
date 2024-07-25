# Water-Quality-Management-using-ML-

Here's an updated README file that includes instructions for using Google Colab and the provided CSV dataset:

---

# Machine Learning Algorithm Implementations

This repository contains implementations of various machine learning algorithms including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, XGBoost, and AdaBoost. Each algorithm is applied to a dataset to demonstrate its functionality and performance.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Logistic Regression](#logistic-regression)
5. [Decision Tree](#decision-tree)
6. [Random Forest](#random-forest)
7. [K-Nearest Neighbors](#k-nearest-neighbors)
8. [Support Vector Machine](#support-vector-machine)
9. [XGBoost](#xgboost)
10. [AdaBoost](#adaboost)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ml-algorithms.git
   ```

## Dataset

The dataset `water_potability.csv` is included in this repository. Ensure it is uploaded to your Google Colab environment for use in the scripts.

## Usage

Each algorithm has its own script, which can be run on Google Colab. Follow these steps:

1. Open Google Colab.
2. Upload the `water_potability.csv` file to your Colab environment.
3. Create a new notebook and follow the instructions for each algorithm.

### Logistic Regression

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### Decision Tree

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### Random Forest

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### K-Nearest Neighbors

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### Support Vector Machine

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### XGBoost

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

### AdaBoost

1. Open a new Colab notebook.
2. Upload the `water_potability.csv` file to the notebook.
3. Copy and paste the following code into a cell and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/water_potability.csv')

# Data preprocessing (replace with actual steps)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize the sections as per your project details.



here is link of all the necessary code
https://colab.research.google.com/drive/1KbBOJ9Zk10hBhWLWrbr4gLSeWG64IdPU?usp=sharing#scrollTo=p6BqHrDiG1BP
