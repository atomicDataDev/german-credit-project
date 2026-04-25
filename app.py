import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Кредитный скоринг", page_icon="🏦", layout="centered")

@st.cache_resource
def load_models():
    model = joblib.load('models/credit_scoring_model.pkl')
    scaler = joblib.load('models/standard_scaler.pkl')
    expected_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

st.title("🏦 Система кредитного скоринга")
st.markdown("Введите данные клиента для оценки вероятности получения кредита")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Параметры кредита")
    amount = st.number_input("Сумма кредита (€)", min_value=100, max_value=20000, value=2500, step=500)
    duration = st.slider("Срок кредита (мес.)", min_value=6, max_value=72, value=12, step=6)
    purpose = st.selectbox("Цель кредита", ["Новый автомобиль", "Б/У автомобиль", "Мебель/Техника", "Образование", "Бизнес", "ТВ/Радио"])

with col2:
    st.subheader("Профиль клиента")
    age = st.number_input("Возраст клиента (лет)", min_value=18, max_value=100, value=30)
    savings = st.selectbox("Сбережения", ["Мало (< 100€)", "Средне", "Много", "Нет сбережений"])
    property_type = st.selectbox("Собственность", ["Недвижимость", "Автомобиль", "Нет собственности"])
st.divider()

st.sidebar.header("⚙️ Настройки риск-менеджмента")
st.sidebar.markdown("Регулировка жесткости скоринга.")

threshold = st.sidebar.slider("Порог одобрения ИИ", min_value=0.40, max_value=0.80, value=0.55, step=0.01)

st.sidebar.divider()
    
st.sidebar.info(f"Текущий порог: **{threshold}**\n\nМодель одобрит кредит, если её уверенность выше этого значения.")    

if st.button("📊 Оценить заемщика", use_container_width=True):

    client_data = {col: 0 for col in expected_columns}

    client_data['credit_amount'] = amount
    client_data['duration'] = duration
    client_data['age'] = age

    if purpose == "Новый автомобиль" and 'purpose_new car' in client_data:
        client_data['purpose_new car'] = 1
    elif purpose == "Радио/ТВ" and 'purpose_radio/tv' in client_data:
        client_data['purpose_radio/tv'] = 1
        
    if savings == "Мало (< 100€)" and 'savings_status_<100' in client_data:
        client_data['savings_status_<100'] = 1    

    if property_type == "Недвижимость" and 'property_magnitude_real estate' in client_data:
        client_data['property_magnitude_real estate'] = 1
    elif property_type == "Нет собственности" and 'property_magnitude_no known property' in client_data:
        client_data['property_magnitude_no known property'] = 1

    df = pd.DataFrame([client_data])
    df_scaled = pd.DataFrame(scaler.transform(df), columns=expected_columns)
    
    prob_good = model.predict_proba(df_scaled)[0, 1]
    
    st.markdown("### Результат анализа:")
    
    st.progress(float(prob_good))
    st.write(f"**Вероятность надежности:** {prob_good * 100:.1f}%")
    
    if prob_good > threshold:
        st.success("✅ **ВЕРДИКТ: КРЕДИТ ОДОБРЕН**")
        st.balloons()
    else:
        st.error("❌ **ВЕРДИКТ: В КРЕДИТЕ ОТКАЗАНО** (Высокий риск дефолта)")