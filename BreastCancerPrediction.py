import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("BreastCancer.csv")
x = data[['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']]
y = data['Class']

# Handle missing values
x_filled = x.fillna('missing')

# Define the correct order for ordinal features (features with natural order)
ordinal_features = {
    'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
                   '45-49', '50-54', '55-59'],
    'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', 
                  '27-29', '30-32', '33-35', '36-39'],
    'deg-malig': ['1', '2', '3'] 
}

# Naive bayes classifier
class Nbayes:
    
    def __init__(self):
        print("Prediction...")
        self.model = CategoricalNB(alpha=1)
        self.trained = False
        self.ordinal_features = ordinal_features
        self.feature_mappings = {}
        self.target_encoder = None
        
    def preprocess_data(self, x, y = None, is_training=True):
        """Preprocess the data with proper ordinal encoding"""
        x_filled = x.fillna('missing')
        x_encoded = x_filled.copy()
        
        for column in x_encoded.columns:
            if column in self.ordinal_features:
                # For ordinal features, use the predefined order
                categories = self.ordinal_features[column]
                if is_training:
                    # Create mapping during training
                    mapping = {category: idx for idx, category in enumerate(categories)}
                    self.feature_mappings[column] = mapping
                else:
                    # Use existing mapping during prediction
                    mapping = self.feature_mappings.get(column, {})
                
                x_encoded[column] = x_encoded[column].map(mapping)
                # Handle any categories not in our predefined list
                x_encoded[column] = x_encoded[column].fillna(len(categories))  # Put unknown at end
                
                if is_training:
                    print(f"{column} encoding: {mapping}")
            else:
                # For nominal features (no natural order)
                if is_training:
                    categories = sorted(x_encoded[column].unique())
                    mapping = {category: idx for idx, category in enumerate(categories)}
                    self.feature_mappings[column] = mapping
                else:
                    mapping = self.feature_mappings.get(column, {})
                
                x_encoded[column] = x_encoded[column].map(mapping)
                # Handle unknown categories in test data
                x_encoded[column] = x_encoded[column].fillna(len(mapping))
                
                if is_training:
                    print(f"{column} encoding: {mapping}")
        
        # Encode target if provided
        if y is not None:
            if is_training:
                from sklearn.preprocessing import LabelEncoder
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y)
                print(f"Target encoding: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
            else:
                y_encoded = self.target_encoder.transform(y)
            return x_encoded, y_encoded
        
        return x_encoded
    
    def fit(self, x_train, y_train):
        x_processed, y_processed = self.preprocess_data(x_train, y_train, is_training=True)
        self.model.fit(x_processed, y_processed)
        self.trained = True
        print("Model trained successfully")

    def predict(self, x_test):
        if not self.trained:
            raise ValueError("Model is not trained yet")
        x_processed = self.preprocess_data(x_test, is_training=False)
        return self.model.predict(x_processed)

    def test(self, x_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet")
        
        y_predict = self.predict(x_test)
        x_processed, y_processed = self.preprocess_data(x_test, y_test, is_training=False)
        
        # Evaluate the model
        accuracy = accuracy_score(y_processed, y_predict)
        print(f"Accuracy: {accuracy:.2f}")

        # Confusion matrix
        cm = confusion_matrix(y_processed, y_predict)
        print("Confusion Matrix:")
        print(cm)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_processed, y_predict, target_names=self.target_encoder.classes_))
        
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Breast Cancer Recurrence Prediction")
        plt.show()
        
        return accuracy
    
    def __del__(self):
        print("Prediction ended") 

def main():
    # Split the data first
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    
    # Create and train the model
    Breast_Cancer_Recurrence = Nbayes()
    Breast_Cancer_Recurrence.fit(x_train, y_train)
    Breast_Cancer_Recurrence.test(x_test, y_test)

if __name__ == "__main__":
    main()