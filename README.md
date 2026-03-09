# Age category classification from digital footprint (multiclass)

Проект: предсказание возрастной категории пользователя по цифровому следу (логи посещений + дополнительные поведенческие признаки).

## Goal
- Построить модель, предсказывающую `age_category` (5 классов) по данным активности пользователя.
- Сформировать витрину признаков из нескольких источников.
- Организовать обучение **без утечек**: split по уникальным `user_id`.
- Оценивать качество по **macro-F1**.

## Data
Источники данных (CSV):
- `ds_s13_users.csv` — целевая переменная `age_category`
- `ds_s13_visits.csv` — логи посещений (категории сайтов, время суток и т.д.)
- `ads_activity.csv`, `surf_depth.csv`, `primary_device.csv`, `cloud_usage.csv` — дополнительные поведенческие признаки

> Данные не хранятся в репозитории. Для запуска положите файлы в папку `data/`.

## Feature engineering
Витрина строится на уровне пользователя, примеры признаков:
- распределения/доли посещений по `website_category`
- распределения по времени суток (`daytime`)
- агрегаты по сессиям (частота, среднее число сессий и т.п.)
- объединение с дополнительными источниками (device / ads_activity / surf_depth / cloud_usage)

Логика сборки признаков вынесена в `feature_builder.py`.

## Models & evaluation
- Baseline: `DummyClassifier`
- Модели: `LogisticRegression`, `SVC` (разные ядра)
- Подбор гиперпараметров: `GridSearchCV`
- Метрики: **macro-F1** (основная), дополнительно macro precision/recall

## Project structure
- `age_category_multiclass.ipynb` — основной ноутбук
- `feature_builder.py` — функции сборки витрины/признаков
- `requirements.txt` — зависимости
- `data/` — папка для датасетов (не хранится в репозитории)

## How to run
1. Положите датасеты в `data/`:
   - `data/ds_s13_users.csv`
   - `data/ds_s13_visits.csv`
   - `data/ads_activity.csv`
   - `data/surf_depth.csv`
   - `data/primary_device.csv`
   - `data/cloud_usage.csv`
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
