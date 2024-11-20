import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
}

# Train and evaluate each classifier
for name, model in classifiers.items():
    print(f"\nTraining {name}...")
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    
    # Save model to separate pickle file
    with open(f'{name.replace(" ", "_").lower()}_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f'{name}: {score * 100:.2f}% of samples were classified correctly.')