import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.model = None
        self.encoder = None
        self.feature_names = None
        self.object_cols = None
        
    def load_data(self):
        """تحميل البيانات من ملف Excel"""
        self.dataset = pd.read_excel(self.data_path)
        return self.dataset
    
    def preprocess_data(self):
        """معالجة البيانات وتنظيفها"""
        # حذف عمود Id
        self.dataset.drop(['Id'], axis=1, inplace=True)
        
        # ملء القيم المفقودة في SalePrice بالمتوسط
        self.dataset['SalePrice'] = self.dataset['SalePrice'].fillna(
            self.dataset['SalePrice'].mean()
        )
        
        # حذف الصفوف التي تحتوي على قيم مفقودة
        self.dataset = self.dataset.dropna()
        
        # تحديد الأعمدة الفئوية
        s = (self.dataset.dtypes == 'object')
        self.object_cols = list(s[s].index)
        
        return self.dataset
    
    def encode_categorical(self):
        """تحويل البيانات الفئوية باستخدام OneHotEncoder"""
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        OH_cols = pd.DataFrame(
            self.encoder.fit_transform(self.dataset[self.object_cols])
        )
        OH_cols.index = self.dataset.index
        OH_cols.columns = self.encoder.get_feature_names_out()
        
        # دمج البيانات المشفرة مع البيانات الأصلية
        df_final = self.dataset.drop(self.object_cols, axis=1)
        df_final = pd.concat([df_final, OH_cols], axis=1)
        
        self.feature_names = df_final.drop(['SalePrice'], axis=1).columns.tolist()
        
        return df_final
    
    def train_model(self, model_type='RandomForest'):
        """تدريب النموذج"""
        # تحضير البيانات
        df_final = self.encode_categorical()
        
        X = df_final.drop(['SalePrice'], axis=1)
        Y = df_final['SalePrice']
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y, train_size=0.8, test_size=0.2, random_state=0
        )
        
        # اختيار النموذج
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=0)
        elif model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif model_type == 'SVR':
            self.model = SVR()
        
        # تدريب النموذج
        self.model.fit(X_train, Y_train)
        
        # التنبؤ والتقييم
        Y_pred = self.model.predict(X_valid)
        mape = mean_absolute_percentage_error(Y_valid, Y_pred)
        
        return {
            'model': self.model,
            'mape': mape,
            'X_train': X_train,
            'X_valid': X_valid,
            'Y_train': Y_train,
            'Y_valid': Y_valid,
            'Y_pred': Y_pred
        }
    
    def predict(self, input_data):
        """التنبؤ بسعر المنزل"""
        if self.model is None:
            raise ValueError("يجب تدريب النموذج أولاً")
        
        # تحويل البيانات المدخلة إلى DataFrame
        input_df = pd.DataFrame([input_data])
        
        # فصل البيانات الفئوية والرقمية
        categorical_data = input_df[self.object_cols]
        numerical_data = input_df.drop(self.object_cols, axis=1)
        
        # تشفير البيانات الفئوية
        encoded_categorical = self.encoder.transform(categorical_data)
        encoded_df = pd.DataFrame(encoded_categorical)
        encoded_df.columns = self.encoder.get_feature_names_out()
        
        # دمج البيانات
        final_input = pd.concat([numerical_data.reset_index(drop=True), encoded_df], axis=1)
        
        # ترتيب الأعمدة حسب ترتيب التدريب
        final_input = final_input[self.feature_names]
        
        # التنبؤ
        prediction = self.model.predict(final_input)[0]
        
        return prediction
    
    def get_feature_importance(self):
        """الحصول على أهمية المتغيرات"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_statistics(self):
        """الحصول على إحصائيات البيانات"""
        numerical_dataset = self.dataset.select_dtypes(include=['number'])
        return {
            'mean': numerical_dataset.mean(),
            'std': numerical_dataset.std(),
            'min': numerical_dataset.min(),
            'max': numerical_dataset.max()
        }
