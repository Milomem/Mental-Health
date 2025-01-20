from pycaret.classification import *
import pandas as pd
import numpy as np

from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

# Drop 'id' column in both datasets
df_train = df.drop(['id'], axis=1)

# Feature Engineering
# Create an interaction term between Age and Work Pressure
df['Age_WorkPressure'] = df['Age'] * df['Work Pressure']
df_test['Age_WorkPressure'] = df_test['Age'] * df_test['Work Pressure']

# Target encoding for categorical features
encoder = TargetEncoder(cols=['City', 'Profession'])
df_train[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(df_train[['City', 'Profession']], df_train["Depression"])
df_test[['City_encoded', 'Profession_encoded']] = encoder.transform(df_test[['City', 'Profession']])

# Dividir df_test em holdout e teste final
df_holdout, df_test_final = train_test_split(df_test, test_size=0.6, random_state=123)

# Define features and target
X_train = df_train.drop('Depression', axis=1)
y_train = df_train['Depression']

# Redefine columns for preprocessing after feature engineering
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

# Initialize setup
clf = setup(data=df_train, target=target_column, session_id=123)

# Compare different models
best_model = compare_models()

metrics = pull()  # Obtém as métricas do PyCaret

# Convert the Pandas DataFrame to a dictionary before logging
metrics_dict = metrics.to_dict(orient='records')  # or 'list', depending on your desired format
# Flatten the metrics dictionary if it's nested
metrics_dict = {k: v for d in metrics_dict for k, v in d.items()}

wandb.log(metrics_dict) # Log the dictionary

# Make predictions on the test set
predictions = predict_model(best_model, data=df_test_final)

predictions.to_csv('previsoes.csv', index=False)
wandb.log_artifact('previsoes.csv', name='previsoes', type='predictions')

# Concatenar os dados de treinamento e holdout
df_retrain = pd.concat([df_train, df_holdout], ignore_index=True)

# Certifique-se que 'Depression' está presente em df_holdout e preencha com 0 se necessário
# Verifique se a coluna 'Depression' existe antes de tentar atribuir valores
if 'Depression' not in df_holdout.columns:
    df_holdout['Depression'] = 0  # Assuming 0 represents 'No Depression' in your case
elif df_holdout['Depression'].isnull().any():
    # Preencha os valores ausentes com 0 se a coluna existir, mas tiver valores ausentes
    df_holdout['Depression'] = df_holdout['Depression'].fillna(0) 

# Redefinir X e y para o novo conjunto de dados
X_retrain = df_retrain.drop('Depression', axis=1)
y_retrain = df_retrain['Depression']

# Reinitialize setup with the concatenated data
clf = setup(data=df_retrain, target=target_column, session_id=123)  


# Retreinar o modelo do zero com os dados combinados
retrained_model = create_model(best_model)  

# Avaliar o modelo
evaluate_model(retrained_model)

# Fazer previsões no conjunto de teste final
final_predictions = predict_model(retrained_model, data=df_test_final)
