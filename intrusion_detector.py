import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():
    # Load the sample dataset
    data = pd.read_csv('sample_network_data.csv')
    X = data[['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'count']]
    y = data['label']
    
    # Train Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'ids_model.joblib')

def predict_sample(sample):
    model = joblib.load('ids_model.joblib')
    prediction = model.predict([sample])
    return "Attack" if prediction[0] == 1 else "Normal"

if __name__ == "__main__":
    train_model()
    test_sample = [0, 0, 491, 0, 2]  # example input
    result = predict_sample(test_sample)
    print(f"The network activity is: {result}")
