# <YOUR_IMPORTS>
import json
from datetime import datetime

import dill
import os
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')


def predict():
    # загружает обученную модель
    with open(f'{path}/data/models/cars_pipe_202302281705.pkl', 'rb') as file:
        model = dill.load(file)

    # делает предсказания для всех объектов в папке data/test
    # объединяет предсказания в один Dataframe и сохраняет их в csv-формате в папку data/predictions.
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for jsonfile in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', jsonfile), 'r') as j:
            data = json.load(j)
            df = pd.DataFrame([data])
            pred = model.predict(df)
            x = {'car_id': df.id, 'pred': pred}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)
    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()


