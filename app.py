import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def train_model():
    dataset = pd.read_excel(os.path.join(BASE_DIR, "HousePricePrediction.xlsx"))
    dataset.drop(['Id'], axis=1, inplace=True)
    dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
    new_dataset = dataset.dropna()

    object_cols = [col for col in new_dataset.columns
                   if new_dataset[col].dtype == 'object' or new_dataset[col].dtype.name == 'str']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    OH_arr = encoder.fit_transform(new_dataset[object_cols])
    OH_cols = pd.DataFrame(OH_arr, index=new_dataset.index, columns=encoder.get_feature_names_out())
    df_final = new_dataset.drop(object_cols, axis=1)
    df_final = pd.concat([df_final, OH_cols], axis=1)

    X = df_final.drop(['SalePrice'], axis=1)
    Y = df_final['SalePrice']

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, train_size=0.8, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    return model, encoder, list(X.columns), object_cols

model, encoder, feature_cols, object_cols = train_model()

st.title("🏠 House Price Prediction")
st.markdown("Fill in the property details below to get an estimated sale price.")

st.subheader("🏗️ Building Info")
col1, col2 = st.columns(2)
with col1:
    MSSubClass = st.selectbox("Building Class (MSSubClass)",
        [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])
    BldgType = st.selectbox("Building Type",
        ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'])
with col2:
    MSZoning = st.selectbox("Zoning Classification",
        ['RL', 'RM', 'FV', 'RH', 'C (all)'])
    OverallCond = st.slider("Overall Condition (1-9)", 1, 9, 5)

st.subheader("📐 Lot Details")
col3, col4 = st.columns(2)
with col3:
    LotArea = st.number_input("Lot Area (sq ft)", min_value=1300, max_value=215245, value=8450, step=100)
with col4:
    LotConfig = st.selectbox("Lot Configuration",
        ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])

st.subheader("🏚️ Exterior & Basement")
col5, col6 = st.columns(2)
with col5:
    Exterior1st = st.selectbox("Exterior Material",
        ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood',
         'BrkFace', 'CemntBd', 'AsbShng', 'Stucco', 'WdShing',
         'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'])
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=6110, value=856, step=10)
with col6:
    BsmtFinSF2 = st.number_input("Finished Basement Type 2 (sq ft)", min_value=0, max_value=1526, value=0, step=10)

st.subheader("📅 Year Info")
col7, col8 = st.columns(2)
with col7:
    YearBuilt = st.number_input("Year Built", min_value=1872, max_value=2010, value=2003, step=1)
with col8:
    YearRemodAdd = st.number_input("Year Remodeled", min_value=1950, max_value=2010, value=2003, step=1)

if st.button("🔮 Predict Sale Price", use_container_width=True):
    input_num = pd.DataFrame([[
        MSSubClass, LotArea, OverallCond, YearBuilt, YearRemodAdd, float(BsmtFinSF2), float(TotalBsmtSF)
    ]], columns=['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF'])

    input_cat = pd.DataFrame([[MSZoning, LotConfig, BldgType, Exterior1st]],
                              columns=object_cols)

    OH_input = pd.DataFrame(encoder.transform(input_cat),
                             columns=encoder.get_feature_names_out())

    input_final = pd.concat([input_num.reset_index(drop=True),
                              OH_input.reset_index(drop=True)], axis=1)
    input_final = input_final.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(input_final)[0]
    prediction = max(0, prediction)

    st.success(f"💰 Estimated Sale Price: **${prediction:,.0f}**")
    st.caption("Powered by Linear Regression · trained on HousePricePrediction.xlsx")
