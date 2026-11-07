from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

st.set_page_config(page_title='Качество вина', layout='wide')
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')

@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X = test[features]
y = test['target']
probabilities = test_proba(model, X)
config = json.loads((BASE / 'config.json').read_text())
threshold = st.sidebar.slider('Порог хорошего вина', 0.0, 1.0, float(config['threshold']), step=0.001, key='threshold')
st.title('Качество вина')
single, quality, batch, weights = st.tabs(['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with single:
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(features):
        with columns[i % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle('Красное вино', value=bool(X[feature].median())))
            else:
                values[feature] = st.number_input(feature, min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].median()), format='%.5f')
    probability = float(model.predict_proba(pd.DataFrame([values], columns=features))[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Рекомендуем в подборку' if probability >= threshold else 'Не рекомендуем в подборку')

with quality:
    prediction = probabilities >= threshold
    precision = metrics.precision_score(y, prediction, zero_division=0)
    recall = metrics.recall_score(y, prediction, zero_division=0)
    f1 = metrics.f1_score(y, prediction, zero_division=0)
    cols = st.columns(3)
    for col, label, value in zip(cols, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(label, f'{value:.3f}')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    matrix = metrics.confusion_matrix(y, prediction, labels=[0, 1])
    metrics.ConfusionMatrixDisplay(matrix, display_labels=['Обычное', 'Хорошее']).plot(ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y, probabilities, ax=axes[1])
    tn, fp, fn, tp = matrix.ravel()
    axes[1].scatter(fp / (fp + tn), recall, color='red', label=f'Порог {threshold:.3f}')
    axes[1].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[2])
    axes[2].axhline(y.mean(), linestyle='--', color='gray')
    axes[2].scatter(recall, precision, color='red', label=f'Порог {threshold:.3f}')
    axes[2].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Если увеличить порог, подборка станет меньше: обычно будет меньше плохих рекомендаций, но больше пропущенных хороших вин. Если уменьшить порог, подборка расширится, но в неё попадёт больше посредственных вин.')

with batch:
    uploaded = st.file_uploader('CSV с 12 признаками вина без target', type=['csv'])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in data.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif data.empty:
                st.error('В файле нет строк с винами.')
            else:
                inputs = data[features].apply(pd.to_numeric, errors='raise')
                if not np.isfinite(inputs.to_numpy(dtype=float)).all():
                    st.error('Все признаки должны быть числами без пропусков и бесконечностей.')
                elif not inputs['is_red'].isin([0, 1]).all():
                    st.error('is_red должен содержать только 0 или 1.')
                else:
                    result = data.copy()
                    result['probability'] = model.predict_proba(inputs)[:, 1]
                    result['prediction'] = (result['probability'] >= threshold).astype(int)
                    st.dataframe(result, hide_index=True)
                    st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeError) as error:
            st.error(f'Не удалось прочитать признаки из CSV: {error}')

with weights:
    coefficients = pd.Series(model[-1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    coefficients.plot.barh(ax=ax, color=np.where(coefficients > 0, 'seagreen', 'indianred'))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент после StandardScaler')
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(coefficients.nlargest(3).index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(coefficients.nsmallest(3).index) + '.')
    st.write('Коэффициенты показывают связь с прогнозом при прочих равных, а не причинное влияние. Связанные признаки могут делить влияние между собой.')
