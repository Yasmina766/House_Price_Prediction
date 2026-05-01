import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from house_price_model import HousePricePredictor
import warnings
warnings.filterwarnings('ignore')

# إعدادات الصفحة
st.set_page_config(
    page_title="تنبؤ أسعار المنازل",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تطبيق نمط مخصص
st.markdown("""
    <style>
        .main {
            padding: 0rem 0rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.title("🏠 نظام تنبؤ أسعار المنازل")
st.markdown("---")

# تحميل البيانات والنموذج
@st.cache_resource
def load_model():
    predictor = HousePricePredictor('HousePricePrediction.xlsx')
    predictor.load_data()
    predictor.preprocess_data()
    
    # تدريب النموذج الافتراضي
    results = predictor.train_model('RandomForest')
    
    return predictor, results

try:
    predictor, results = load_model()
    dataset = predictor.dataset
    
    # القائمة الجانبية
    st.sidebar.title("⚙️ الخيارات")
    
    # اختيار الصفحة
    page = st.sidebar.radio(
        "اختر الصفحة:",
        ["🔮 التنبؤ", "📊 تحليل البيانات", "📈 أداء النموذج", "ℹ️ معلومات"]
    )
    
    # ============ صفحة التنبؤ ============
    if page == "🔮 التنبؤ":
        st.header("التنبؤ بسعر المنزل")
        st.markdown("أدخل معلومات المنزل للحصول على تنبؤ بسعره")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("البيانات الرقمية")
            
            ms_sub_class = st.slider(
                "نوع المسكن (MSSubClass)",
                min_value=int(dataset['MSSubClass'].min()),
                max_value=int(dataset['MSSubClass'].max()),
                value=int(dataset['MSSubClass'].mean())
            )
            
            lot_area = st.slider(
                "مساحة الأرض بالقدم المربع (LotArea)",
                min_value=int(dataset['LotArea'].min()),
                max_value=int(dataset['LotArea'].max()),
                value=int(dataset['LotArea'].mean())
            )
            
            overall_cond = st.slider(
                "الحالة العامة للمنزل (OverallCond)",
                min_value=int(dataset['OverallCond'].min()),
                max_value=int(dataset['OverallCond'].max()),
                value=int(dataset['OverallCond'].mean())
            )
            
            year_built = st.slider(
                "سنة البناء (YearBuilt)",
                min_value=int(dataset['YearBuilt'].min()),
                max_value=int(dataset['YearBuilt'].max()),
                value=int(dataset['YearBuilt'].mean())
            )
        
        with col2:
            st.subheader("البيانات الرقمية (تابع)")
            
            year_remod_add = st.slider(
                "سنة التجديد (YearRemodAdd)",
                min_value=int(dataset['YearRemodAdd'].min()),
                max_value=int(dataset['YearRemodAdd'].max()),
                value=int(dataset['YearRemodAdd'].mean())
            )
            
            bsmt_fin_sf2 = st.slider(
                "مساحة الطابق السفلي النوع 2 (BsmtFinSF2)",
                min_value=int(dataset['BsmtFinSF2'].min()),
                max_value=int(dataset['BsmtFinSF2'].max()),
                value=int(dataset['BsmtFinSF2'].mean())
            )
            
            total_bsmt_sf = st.slider(
                "إجمالي مساحة الطابق السفلي (TotalBsmtSF)",
                min_value=int(dataset['TotalBsmtSF'].min()),
                max_value=int(dataset['TotalBsmtSF'].max()),
                value=int(dataset['TotalBsmtSF'].mean())
            )
        
        st.subheader("البيانات الفئوية")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            ms_zoning = st.selectbox(
                "تصنيف المنطقة (MSZoning)",
                options=dataset['MSZoning'].unique()
            )
        
        with col4:
            lot_config = st.selectbox(
                "تكوين الأرض (LotConfig)",
                options=dataset['LotConfig'].unique()
            )
        
        with col5:
            bldg_type = st.selectbox(
                "نوع المبنى (BldgType)",
                options=dataset['BldgType'].unique()
            )
        
        col6, col7 = st.columns(2)
        
        with col6:
            exterior1st = st.selectbox(
                "الطلاء الخارجي (Exterior1st)",
                options=dataset['Exterior1st'].unique()
            )
        
        with col7:
            model_choice = st.selectbox(
                "اختر نموذج التنبؤ",
                options=['RandomForest', 'LinearRegression', 'SVR']
            )
        
        # زر التنبؤ
        if st.button("🎯 تنبؤ بالسعر", use_container_width=True):
            # إعادة تدريب النموذج إذا تم اختيار نموذج مختلف
            if model_choice != 'RandomForest':
                results = predictor.train_model(model_choice)
            
            # تحضير البيانات المدخلة
            input_data = {
                'MSSubClass': ms_sub_class,
                'MSZoning': ms_zoning,
                'LotArea': lot_area,
                'LotConfig': lot_config,
                'BldgType': bldg_type,
                'OverallCond': overall_cond,
                'YearBuilt': year_built,
                'YearRemodAdd': year_remod_add,
                'Exterior1st': exterior1st,
                'BsmtFinSF2': bsmt_fin_sf2,
                'TotalBsmtSF': total_bsmt_sf
            }
            
            # التنبؤ
            try:
                prediction = predictor.predict(input_data)
                
                # عرض النتيجة
                st.success("✅ تم التنبؤ بنجاح!")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        "السعر المتنبأ به",
                        f"${prediction:,.2f}",
                        delta=None
                    )
                
                with col_result2:
                    st.metric(
                        "دقة النموذج (MAPE)",
                        f"{results['mape']*100:.2f}%",
                        delta=None
                    )
                
                # عرض معلومات إضافية
                st.info(f"""
                    📋 **ملخص التنبؤ:**
                    - **نوع المسكن:** {ms_sub_class}
                    - **مساحة الأرض:** {lot_area:,} قدم مربع
                    - **سنة البناء:** {year_built}
                    - **الحالة العامة:** {overall_cond}/10
                    - **نموذج التنبؤ:** {model_choice}
                """)
                
            except Exception as e:
                st.error(f"❌ حدث خطأ في التنبؤ: {str(e)}")
    
    # ============ صفحة تحليل البيانات ============
    elif page == "📊 تحليل البيانات":
        st.header("تحليل البيانات")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 الإحصائيات", "🔗 الارتباطات", "📊 التوزيع", "ℹ️ معلومات البيانات"])
        
        with tab1:
            st.subheader("الإحصائيات الوصفية")
            
            numerical_dataset = dataset.select_dtypes(include=['number'])
            stats_df = numerical_dataset.describe().T
            
            st.dataframe(stats_df, use_container_width=True)
        
        with tab2:
            st.subheader("مصفوفة الارتباط")
            
            numerical_dataset = dataset.select_dtypes(include=['number'])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                numerical_dataset.corr(),
                cmap='BrBG',
                fmt='.2f',
                linewidths=2,
                annot=True,
                ax=ax,
                cbar_kws={'label': 'معامل الارتباط'}
            )
            ax.set_title('مصفوفة الارتباط بين المتغيرات الرقمية', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            st.pyplot(fig)
        
        with tab3:
            st.subheader("توزيع أسعار المنازل")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(dataset['SalePrice'], bins=50, color='skyblue', edgecolor='black')
            ax.set_xlabel('السعر ($)')
            ax.set_ylabel('التكرار')
            ax.set_title('توزيع أسعار المنازل', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("متوسط السعر", f"${dataset['SalePrice'].mean():,.0f}")
            
            with col2:
                st.metric("الوسيط", f"${dataset['SalePrice'].median():,.0f}")
            
            with col3:
                st.metric("أقل سعر", f"${dataset['SalePrice'].min():,.0f}")
            
            with col4:
                st.metric("أعلى سعر", f"${dataset['SalePrice'].max():,.0f}")
        
        with tab4:
            st.subheader("معلومات البيانات")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("عدد الصفوف", len(dataset))
            
            with col2:
                st.metric("عدد الأعمدة", len(dataset.columns))
            
            with col3:
                st.metric("عدد القيم المفقودة", dataset.isnull().sum().sum())
            
            st.subheader("أنواع البيانات")
            
            dtype_df = pd.DataFrame({
                'العمود': dataset.columns,
                'النوع': dataset.dtypes.astype(str)
            })
            
            st.dataframe(dtype_df, use_container_width=True)
    
    # ============ صفحة أداء النموذج ============
    elif page == "📈 أداء النموذج":
        st.header("أداء النموذج")
        
        tab1, tab2, tab3 = st.tabs(["📊 الأداء", "🎯 التنبؤات", "⚙️ المقارنة"])
        
        with tab1:
            st.subheader("مقاييس الأداء")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "MAPE (Mean Absolute Percentage Error)",
                    f"{results['mape']*100:.2f}%"
                )
            
            with col2:
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                mae = mean_absolute_error(results['Y_valid'], results['Y_pred'])
                st.metric("MAE", f"${mae:,.0f}")
            
            with col3:
                rmse = np.sqrt(mean_squared_error(results['Y_valid'], results['Y_pred']))
                st.metric("RMSE", f"${rmse:,.0f}")
        
        with tab2:
            st.subheader("التنبؤات مقابل القيم الفعلية")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.scatter(results['Y_valid'], results['Y_pred'], alpha=0.5, s=30)
            
            # خط مثالي
            min_val = min(results['Y_valid'].min(), results['Y_pred'].min())
            max_val = max(results['Y_valid'].max(), results['Y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='التنبؤ المثالي')
            
            ax.set_xlabel('القيم الفعلية ($)')
            ax.set_ylabel('القيم المتنبأ بها ($)')
            ax.set_title('مقارنة التنبؤات بالقيم الفعلية', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab3:
            st.subheader("مقارنة النماذج المختلفة")
            
            models_to_compare = ['RandomForest', 'LinearRegression', 'SVR']
            comparison_results = {}
            
            for model_name in models_to_compare:
                try:
                    model_results = predictor.train_model(model_name)
                    comparison_results[model_name] = model_results['mape'] * 100
                except:
                    comparison_results[model_name] = None
            
            comparison_df = pd.DataFrame({
                'النموذج': list(comparison_results.keys()),
                'MAPE (%)': list(comparison_results.values())
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # رسم بياني للمقارنة
            fig, ax = plt.subplots(figsize=(10, 5))
            
            valid_models = [k for k, v in comparison_results.items() if v is not None]
            valid_mapes = [v for v in comparison_results.values() if v is not None]
            
            ax.bar(valid_models, valid_mapes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('MAPE (%)')
            ax.set_title('مقارنة أداء النماذج المختلفة', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
    
    # ============ صفحة المعلومات ============
    elif page == "ℹ️ معلومات":
        st.header("معلومات عن التطبيق")
        
        st.markdown("""
        ## 🏠 نظام تنبؤ أسعار المنازل
        
        ### 📝 الوصف
        هذا التطبيق يستخدم نماذج التعلم الآلي للتنبؤ بأسعار المنازل بناءً على مجموعة من الخصائص المختلفة.
        
        ### 🎯 الميزات
        - **التنبؤ الفوري:** احصل على تنبؤ بسعر المنزل في الحال
        - **تحليل البيانات:** استكشف البيانات والعلاقات بين المتغيرات
        - **مقارنة النماذج:** قارن بين نماذج التعلم الآلي المختلفة
        - **واجهة سهلة الاستخدام:** تصميم بديهي وسهل الاستخدام
        
        ### 🤖 النماذج المستخدمة
        1. **Random Forest Regressor:** نموذج قائم على الغابات العشوائية
        2. **Linear Regression:** نموذج الانحدار الخطي
        3. **Support Vector Regression (SVR):** آلات المتجهات الداعمة
        
        ### 📊 البيانات
        - **عدد السجلات:** 2,919 منزل
        - **عدد المتغيرات:** 13 متغير
        - **المتغيرات الرقمية:** 9 متغيرات
        - **المتغيرات الفئوية:** 4 متغيرات
        
        ### 📈 مقاييس الأداء
        - **MAPE:** Mean Absolute Percentage Error
        - **MAE:** Mean Absolute Error
        - **RMSE:** Root Mean Squared Error
        
        ### 🔧 التقنيات المستخدمة
        - **Python:** لغة البرمجة الأساسية
        - **Streamlit:** لإنشاء الواجهة التفاعلية
        - **Scikit-learn:** لنماذج التعلم الآلي
        - **Pandas:** لمعالجة البيانات
        - **Matplotlib & Seaborn:** للرسوم البيانية
        
        ### 💡 كيفية الاستخدام
        1. اذهب إلى صفحة "التنبؤ"
        2. أدخل معلومات المنزل
        3. اختر نموذج التنبؤ
        4. انقر على "تنبؤ بالسعر"
        5. احصل على النتيجة الفورية
        
        ### 📧 التطوير والدعم
        تم تطوير هذا التطبيق باستخدام أحدث تقنيات التعلم الآلي والبرمجة.
        """)
        
        st.markdown("---")
        
        st.info("""
        **ملاحظة:** هذا التطبيق للأغراض التعليمية والتجريبية. 
        قد تختلف التنبؤات عن الأسعار الفعلية في السوق.
        """)

except Exception as e:
    st.error(f"❌ حدث خطأ في تحميل التطبيق: {str(e)}")
    st.info("تأكد من وجود ملف `HousePricePrediction.xlsx` في نفس المجلد")
