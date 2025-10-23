
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder #to deal with the categorial data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt

#Data Processing
#Upload, Read and encode data
data = pd.read_csv("weather.csv")
x = data[['Outlook','Temperature','Humidity','Windy']]
y = data['Play']


label_encoders = {}
x_encoded = pd.DataFrame()

for column in x.columns:
    le = LabelEncoder()
    x_encoded[column] = le.fit_transform(x[column])
    label_encoders[column] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


#Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size = 0.2, random_state = 42, stratify=y_encoded)


#Naive bayes classifier
class Nbayes:
    
    def __init__ (self, data):
        print("Prediction...")
        self.model = CategoricalNB(alpha = 1)
        self.trained = False
        self.data = data
        self.label_encoders = label_encoders
        self.target_encoder = le_target
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.trained = True

    def predict(self, x_test):
        if not self.trained:
            raise ValueError("Model is not trained yet")
        return self.model.predict(x_test)

    def test(self, x_test, y_test):
        if not self.trained:
            raise ValueError("Model is not trained yet")
        y_predict = self.predict(x_test)
        
        #Accuracy
        acc = accuracy_score(y_test, y_predict)
        print("Accuracy:", acc)
        
        #Confusion matrix
        cm = confusion_matrix(y_test, y_predict)
        print("Confusion Matrix:\n", cm)
        
        #Classification report
        target_names = self.target_encoder.classes_
        report = classification_report(y_test, y_predict, target_names=target_names)

        print("Classification Report:\n", report)
        
        #Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        return acc
    
    def __del__(self):
        print("Prediction ended") 

def main():
    weather = Nbayes(data)
    weather.fit(x_train, y_train)
    weather.test(x_test, y_test)

if __name__ == "__main__":
    main()
    