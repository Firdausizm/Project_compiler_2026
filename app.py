import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Dashboard Transjakarta", page_icon="🚌", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('dfTransjakarta_cleaned.csv')
    return df

@st.cache_resource
def load_model():
    return joblib.load('density_model.pkl')

st.sidebar.title("🚌 Menu Utama")
menu = st.sidebar.radio("Navigasi", ["Ikhtisar Data & EDA", "Prediksi Kepadatan Halte"])

st.title("Dashboard Transjakarta Analitik & Prediksi")

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

if menu == "Ikhtisar Data & EDA":
    st.header("Ikhtisar Perjalanan")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Transaksi</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Pendapatan</div><div class="metric-value">Rp {df["payAmount"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Durasi (menit)</div><div class="metric-value">{df["trip_duration_mins"].mean():.1f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Usia Penumpang</div><div class="metric-value">{df["age"].mean():.1f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # EDA Charts
    st.subheader("Tren Penumpang Berdasarkan Jam")
    hourly_counts = df['tapInHour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['Jam', 'Jumlah Penumpang']
    fig1 = px.line(hourly_counts, x='Jam', y='Jumlah Penumpang', markers=True, title='Kepadatan Tap-In Berdasarkan Jam dalam Sehari')
    st.plotly_chart(fig1, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Demografi Gender")
        gender_counts = df['payCardSex'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Jumlah']
        fig2 = px.pie(gender_counts, names='Gender', values='Jumlah', hole=0.4, title='Distribusi Gender')
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_chart2:
        st.subheader("Penggunaan Bank Penerbit Kartu")
        bank_counts = df['payCardBank'].value_counts().reset_index()
        bank_counts.columns = ['Bank', 'Jumlah']
        fig3 = px.bar(bank_counts, x='Bank', y='Jumlah', title='Popularitas Bank Penerbit Kartu', color='Bank')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 10 Halte Tap-In Tersibuk")
    top_stops = df['tapInStopsName'].value_counts().head(10).reset_index()
    top_stops.columns = ['Nama Halte', 'Jumlah Penumpang']
    fig4 = px.bar(top_stops, x='Jumlah Penumpang', y='Nama Halte', orientation='h', title='Halte dengan Tap-In Terbanyak')
    fig4.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

elif menu == "Prediksi Kepadatan Halte":
    st.header("Prediksi Kepadatan Penumpang di Halte")
    st.write("Gunakan Machine Learning (Random Forest) untuk memprediksi jumlah penumpang yang melakukan Tap-In di halte tertentu pada hari dan jam yang spesifik.")
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
        
    # User Inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        halte_list = sorted(df['tapInStopsName'].dropna().unique())
        selected_stop = st.selectbox("Pilih Halte", halte_list)
        
    with col2:
        day_mapping = {
            "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, 
            "Jumat": 4, "Sabtu": 5, "Minggu": 6
        }
        selected_day = st.selectbox("Pilih Hari", list(day_mapping.keys()))
        
    with col3:
        selected_hour = st.slider("Pilih Jam", 0, 23, 8)
        
    if st.button("Prediksi Kepadatan", type="primary"):
        input_data = pd.DataFrame({
            'tapInStopsName': [selected_stop],
            'tapInDayOfWeek': [day_mapping[selected_day]],
            'tapInHour': [selected_hour]
        })
        
        with st.spinner('Menghitung prediksi...'):
            prediction = model.predict(input_data)[0]
            
        st.success(f"Berdasarkan model prediktif, diperkirakan akan ada **{int(round(prediction))} penumpang** yang melakukan Tap-In di Halte **{selected_stop}** pada hari **{selected_day}** jam **{selected_hour}:00**.")
        
        # Contextual visualization
        st.subheader("Konteks Historis")
        historical = df[(df['tapInStopsName'] == selected_stop) & (df['tapInDayOfWeek'] == day_mapping[selected_day])]
        if not historical.empty:
            hist_agg = historical.groupby('tapInHour').size().reset_index(name='Jumlah')
            fig_hist = px.bar(hist_agg, x='tapInHour', y='Jumlah', title=f'Historis Kepadatan Halte {selected_stop} pada Hari {selected_day}')
            fig_hist.add_vline(x=selected_hour, line_width=3, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Tidak ada data historis yang cukup untuk menampilkan grafik konteks pada halte dan hari ini.")
