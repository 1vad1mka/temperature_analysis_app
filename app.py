# Импорт библиотек
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

pd.options.plotting.backend = "plotly"


# Устанавливаем plotly в качестве рисунков в pandas
pd.options.plotting.backend = "plotly"


def plot_ts(data, color_col=None):
  if color_col:
    fig = px.line(data, x=data.index, y="temperature", color=color_col)
    fig.show()
  else:
    fig = px.line(data, x=data.index, y="temperature")
    #fig.show()
    return fig
  

def plot_smoothed_ts(data):
  # Создание фигуры
  fig = go.Figure()

  # Добавление временного ряда
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=data['temperature'], mode='lines', name='Временной ряд'))

  # Добавление скользящего среднего
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=data['rolling_avg_temperature'], mode='lines', name='Скользящее среднее'))

  # Настройки графика
  fig.update_layout(title='Температура и скользящее среднее температуры',
                    xaxis_title='Дата',
                    yaxis_title='Температура')

  # Показать график
  #fig.show()
  return fig


def plot_smoothed_with_interval(data):
  # Создание фигуры
  fig = go.Figure()

  # Добавление временного ряда
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=data['temperature'], mode='lines', name='Временной ряд'))

  # Добавление скользящего среднего
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=data['rolling_avg_temperature'], mode='lines', name='Скользящее среднее'))

  # Добавление тени для интервала ±2 стандартных отклонения
  fig.add_trace(go.Scatter(
      x=data.sort_index().index,
      y=data['rolling_avg_temperature'] + 2 * data['rolling_std_temperature'],
      fill=None,
      mode='lines',
      line=dict(color='lightyellow'),
      name='+2 стандартных отклонения'
  ))

  fig.add_trace(go.Scatter(
      x=data.sort_index().index,
      y=data['rolling_avg_temperature'] - 2 * data['rolling_std_temperature'],
      fill='tonexty',  # Заполнение области между линиями
      mode='lines',
      line=dict(color='lightyellow'),
      name='-2 стандартных отклонения'
  ))

  # Настройки графика
  fig.update_layout(title='Температура и скользящее среднее температуры с интервалом ±2 стандартных отклонения',
                    xaxis_title='Дата',
                    yaxis_title='Температура')

  # Показать график
  #fig.show()
  return fig


def normal_interval(row):
  lower_interval = row['rolling_avg_temperature'] - 2*row['rolling_std_temperature']
  upper_interval = row['rolling_avg_temperature'] + 2*row['rolling_std_temperature']

  return (lower_interval, upper_interval)



def is_normal(row):
  if row['temperature'] >= row['normal_interval'][0] and row['temperature'] <= row['normal_interval'][1]:
    return True

  else:
    return False
  

def plot_ts_trend(data, trend):

  # Создание фигуры
  fig = go.Figure()

  # Добавление временного ряда
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=data, mode='lines', name='Временной ряд'))

  # Добавление тренда
  fig.add_trace(go.Scatter(x=data.sort_index().index, y=trend, mode='lines', name='тренд'))

  # Настройки графика
  fig.update_layout(title='Температура и тренд',
                    xaxis_title='Дата',
                    yaxis_title='Температура')

  # Показать график
  #fig.show()
  return fig



def PredictProphet(data, forecast_period=365):
  """
  data:
  - timestamp (as index)
  - temperature
  """

  prophet_data = data.reset_index().rename(columns={'timestamp': 'ds', 'temperature': 'y'})
  model_prophet = Prophet()
  model_prophet.fit(prophet_data)

  future = model_prophet.make_future_dataframe(periods=forecast_period)
  predict_index = pd.date_range(start=pd.to_datetime(max(data.index)), periods=forecast_period, freq='D')

  forecast_prophet = model_prophet.predict(future)
  pred_prophet = forecast_prophet['yhat'][len(data):]
  pred_prophet.index = predict_index

  return data, pred_prophet


def plot_prophet_predictions(data, pred_prophet):
  # Создание графика
  fig = go.Figure()

  # Добавление реальных значений
  fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Реальные значения', line=dict(color='blue')))


  # Добавление предсказанных значений
  fig.add_trace(go.Scatter(x=pred_prophet.index, y=pred_prophet, mode='lines', name='Предсказанные значения', line=dict(color='red')))

  # Настройка графика
  fig.update_layout(title='Предсказание температуры с использованием Prophet',
                    xaxis_title='Дата',
                    yaxis_title='Температура',
                    legend=dict(x=0, y=1))

  # Показать график
  #fig.show()
  return fig





def get_coordinates(city_name, token):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit={1}&appid={token}"

    try:
        with requests.Session() as session:
            with session.get(url) as response:
                response_data = response.json()

    except Exception as e:
        print(f'Произошла ошибка. Детали: {e}')

    else:
        return response_data
    

def get_current_weather(lat, lon, token, units='metric'):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={token}&units={units}"

    try:
        with requests.Session() as session:
            with session.get(url) as response:
                response_data = response.json()

    except Exception as e:
        print(f'Произошла ошибка. Детали: {e}')

    else:
        return response_data
    

# Функция для отправки синхронных API-запросов
def API_request_sync(API_key, city_name):
  # Получаем информацию о координатах
  response = get_coordinates(city_name, token=API_key)
  name, lat, lon = response[0]['name'], response[0]['lat'], response[0]['lon']

  # Получаем информацию о погоде
  response = get_current_weather(lat, lon, API_key)
  temp, temp_min, temp_max, date = response['main']['temp'], response['main']['temp_min'], response['main']['temp_max'], response['dt']
  date = datetime.utcfromtimestamp(int(date)).strftime('%Y-%m-%d')

  return temp, temp_min, temp_max, date
    


# Определяем сезон по дате
def get_season(date):
  month = int(date.split('-')[1])
  if month in [12, 1, 2]:
    return "winter"
  elif month in [3, 4, 5]:
    return "spring"
  elif month in [6, 7, 8]:
    return "summer"
  elif month in [9, 10, 11]:
    return "autumn"
  

def analysis_sync(data, API_key, city_name):
  """
  Функция для синхронного анализа данных. Принимает данные для одного города с колонками и колонками:
  1. `city`
  2. `timestamp`
  3. `temperature`
  4. `season`

  Если данные не в таком формате, то я пытаюсь привести данные к подобному формату

  Функция возвращает
  1. название города
  2. данные со скользящим средним
  3. наклон
  4. аномалии
  """

  # Выбираем необходимые данные и приводим датасет к нужному виду
  try:
    columns_to_drop = []
    for col in data.columns:
      if col not in ['city', 'timestamp', 'temperature', 'season']:
        columns_to_drop.append(col)

    data = data.drop(columns_to_drop, axis=1)

    if 'city' in data.columns:
      data = data.drop(['city'], axis=1)

    if 'timestamp' in data.columns:
      data = data.set_index('timestamp')
      data.index = pd.to_datetime(data.index)

  except Exception as e:
    print(f'Произошла ошибка: {e}')


  # Считаем среднее и стандартное отклонение.
  data_rolling = data.sort_index().rolling(window='30D')['temperature'].agg(['mean', 'std'])
  data_rolling = data_rolling.rename({'mean': 'rolling_avg_temperature', 'std': 'rolling_std_temperature'}, axis=1)
  data = pd.concat([data, data_rolling], axis=1)


  # Считаем нормальный интервал
  data['normal_interval'] = data.apply(normal_interval, axis=1)
  # Определяем являются ли данные нормальными
  data['is_normal'] = data.apply(is_normal, axis=1)

  # Выделим аномальные значения
  data_anomaly = data.query('is_normal==False')

  # Профили сезона. Потом нужно будет передать в функцию для одного города
  season_profile = data.groupby('season')['temperature'].agg(['mean', 'std', 'min', 'max'])
  season_profile['normal_min'] = season_profile.apply(lambda row: row['mean'] - 2*row['std'], axis=1)
  season_profile['normal_max'] = season_profile.apply(lambda row: row['mean'] + 2*row['std'], axis=1)

  # Выявление трендов
  # Получаем признаки для обучения и целевую переменную
  data_regression = data['temperature']

  X = data_regression.reset_index().timestamp.apply(lambda x: pd.Timestamp(x).toordinal())
  y = data_regression.reset_index().temperature
  # Приводим данные к нужному типу
  X = X.to_numpy().reshape(-1, 1)
  y = y.to_numpy()
  # Обучаем модель
  model = LinearRegression()
  model.fit(X, y)
  # Получаем уклон (коэффициент наклона)
  trend = model.predict(X)
  slope = model.coef_[0]


  # Отправляем запрос к API
  # Получаем информацию о координатах
  try:
    temp, temp_min, temp_max, date = API_request_sync(API_key, city_name)

  except:
     raise Exception(f"Wrong API key: {'{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}'}")

  # Определяем сезон по дате
  season = get_season(date)

  # Проверяем температуру на нормальность
  normal_temp = None

  if temp >= season_profile.loc[season, 'normal_min'] and temp <= season_profile.loc[season, 'normal_max']:
    normal_temp = True
  else:
    normal_temp = False

  # рисуем необходимые графики
  fig_ts = plot_ts(data)
  fig_ts_smoothed = plot_smoothed_ts(data)
  fig_smoothed_interval = plot_smoothed_with_interval(data)
  fig_ts_trend = plot_ts_trend(data_regression, trend=trend)


  # Заключим все графики в словарь, чтобы позже было легко их доставать
  plots = {'time_series': fig_ts, 'time_series_smooth': fig_ts_smoothed, 'time_series_int': fig_smoothed_interval, 'time_series_trend': fig_ts_trend}

  return city_name, data, season_profile, slope, data_anomaly, plots, normal_temp, temp, season


cities_list = ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney', 'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles', 'Singapore', 'Mumbai', 'Cairo', 'Mexico City']


# Зададим заголовок
st.title("Анализ нормальности температуры в городах")

# Здесь пользователь может загрузить данные
st.header("Загрузка данных")

uploaded_file = st.file_uploader("Загрузите CSV-файл с историческими данными о температуре в городе", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:")
    st.dataframe(data)

    city_name = st.selectbox("Выберите город для анализа", options=cities_list)
    API_key = st.text_input("Введите свой API-ключ")


    # Отобразим основные описательные статистики
    # Средняя температура по годам

    start_analysis_button = st.button("Начать анализ", type="primary")

    if start_analysis_button:
        st.header("Анализ температуры")

        city_name, data, season_profile, slope, data_anomaly, plots, normal_temp, temp, season = analysis_sync(data, API_key, city_name=city_name)

        # Основная информация о температуре
        st.write(f"Информация о температуре в городе {city_name}:")
        st.markdown(f"- **Время года** : {season}")
        st.markdown(f"- **Текущая температура** : {temp}")
        st.markdown(f"- **Наклон линии тренда** : {slope}")


        # Нормальна ли температура?
        if normal_temp:
            st.write(f"Температура {temp} для города '{city_name}' во время года '{season}' является нормальной!")
        else:
           st.write(f"Температура {temp} для города '{city_name}' во время года '{season}' является аномальной!")


        st.write("Основные статистики температуры:")
        st.dataframe(data['temperature'].describe())


        st.write(f"Профиль сезонов для города {city_name}:")
        st.dataframe(season_profile)

        st.header("Графики")

        st.write("Динамика температуры за исторический период:")
        st.plotly_chart(plots['time_series'])

        st.write("Сгаженная динамика температуры за исторический период:")
        st.plotly_chart(plots['time_series_smooth'])

        st.write("Интервал 'Нормальной' исторической темпеaратуры:")
        st.plotly_chart(plots['time_series_int'])

        st.write("Тренд температуры за исторический период:")
        st.plotly_chart(plots['time_series_trend'])


        # Изобразим распределением температур
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x="temperature", y="season", palette="Set2", linewidth=1.5)
        plt.title("Распределение температуры по сезонам", fontsize=16)
        plt.xlabel("Температура (°C)", fontsize=14)
        plt.ylabel("Сезон", fontsize=14)
        st.pyplot(plt)


        st.header("Предсказание температуры")
        #prophet_period = st.number_input("Выберите горизонт предсказания в днях", 7, 1000)

        data_to_prophet = data['temperature']
        data_to_prophet, pred_prophet = PredictProphet(data_to_prophet, forecast_period=730)
        st.write("Предсказание температуры:")
        st.plotly_chart(plot_prophet_predictions(data_to_prophet, pred_prophet))

else:
    st.write("Пожалуйста, загрузите CSV-файл.")