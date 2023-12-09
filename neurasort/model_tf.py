from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.metrics import F1Score

def create_model(
    input_dim=13,
    num_classes=4,
    optimizer='adagrad',
    loss='categorical_crossentropy',
    hidden_layers_dims = [
        32, 64, 64
    ],
    metrics=[F1Score("weighted")]
):
    model = Sequential()
    model.add(Dense(hidden_layers_dims[0], input_dim=input_dim))
    model.add(Activation('relu'))
    for i in hidden_layers_dims[1:]:
        model.add(Dense(i))
        model.add(Activation('relu'))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

if __name__ == "__main__":
    from scikeras.wrappers import KerasClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline 
    from sklearn.preprocessing import StandardScaler
    
    from keras.utils import to_categorical
    X, y = make_classification(
        n_samples=1000, # 1000 observations 
        n_features=5, # 5 total features
        n_informative=3, # 3 'useful' features
        n_classes=4, # binary target/label 
        random_state=999 # if you want the same results as mine
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(f"{y=}")
    y_train_one_hot = to_categorical(y_train, num_classes=4)
    y_test_one_hot = to_categorical(y_test, num_classes=4)
    print(f'{y_train_one_hot=}')
    clf_tf = KerasClassifier(build_fn=create_model, verbose=0)
    clf_tf.fit(X_train, y_train_one_hot, validation_data=[X_test, y_test_one_hot], epochs=10)
    print(clf_tf.history_.keys())
    print(clf_tf.history_["val_f1_score"])
    
    scaler = StandardScaler()
    
    pipe = Pipeline(
        [
            ("scaler", scaler),
            ("clf", clf_tf)
        ]
    )
    
    pipe.fit(
        X_train,
        y_train_one_hot,
        clf__validation_data = [X_test, y_test_one_hot],
        clf__epochs = 10,
    )
    print(f"{pipe.steps[-1][-1].history_['val_f1_score']=}")
    
    