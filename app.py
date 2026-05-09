import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Transjakarta", page_icon="🚌", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 13px;
        color: #aaa;
        margin-bottom: 6px;
    }
    .insight-box {
        background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #6C63FF;
        margin-bottom: 12px;
        color: #ddd;
    }
    .insight-title {
        font-size: 15px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

DARK_TEMPLATE = "plotly_dark"
COLOR_PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#F9C74F", "#4CC9F0", "#FF9F43"]

GEN_MAP = {
    'Gen Z (1997-2012)': (1997, 2012),
    'Gen Y (1981-1996)': (1981, 1996),
    'Gen X (1965-1980)': (1965, 1980),
}

ROUTES = [
    {'label': 'Pal Putih → Tegalan', 'tap_in': 'Pal Putih', 'tap_out': 'Tegalan'},
    {'label': 'Gg. Kunir II → Simpang Kunir Kemukus', 'tap_in': 'Gg. Kunir II', 'tap_out': 'Simpang Kunir Kemukus'},
]

@st.cache_data
def load_data():
    df = pd.read_csv('dfTransjakarta_cleaned.csv')
    df['tapInTime'] = pd.to_datetime(df['tapInTime'], errors='coerce')
    df['tapOutTime'] = pd.to_datetime(df['tapOutTime'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    return joblib.load('density_model.pkl')

def assign_gen(year):
    for name, (lo, hi) in GEN_MAP.items():
        if lo <= year <= hi:
            return name
    return None

# --- Sidebar ---
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio("Navigasi", [
    "Ikhtisar Data & EDA",
    "Business Insights",
    "Prediksi Kepadatan Halte"
])

st.title("Dashboard Transjakarta Analitik & Prediksi")

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

if menu == "Ikhtisar Data & EDA":
    st.header("Ikhtisar Perjalanan")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Transaksi</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Pendapatan</div><div class="metric-value">Rp {df["payAmount"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with col3:
        avg_dur = df["trip_duration_mins"].mean()
        dur_str = f"{avg_dur:.1f}" if not pd.isna(avg_dur) else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Durasi (menit)</div><div class="metric-value">{dur_str}</div></div>', unsafe_allow_html=True)
    with col4:
        avg_age = df["age"].mean()
        age_str = f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Usia Penumpang</div><div class="metric-value">{age_str}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tren penumpang per jam
    st.subheader("Tren Penumpang Berdasarkan Jam")
    hourly = df['tapInHour'].value_counts().sort_index().reset_index()
    hourly.columns = ['Jam', 'Jumlah Penumpang']
    fig1 = px.line(hourly, x='Jam', y='Jumlah Penumpang', markers=True,
                   title='Kepadatan Tap-In Berdasarkan Jam',
                   template=DARK_TEMPLATE, color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Demografi Gender")
        gender = df['payCardSex'].value_counts().reset_index()
        gender.columns = ['Gender', 'Jumlah']
        fig2 = px.pie(gender, names='Gender', values='Jumlah', hole=0.45,
                      title='Distribusi Gender', template=DARK_TEMPLATE,
                      color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.subheader("Penggunaan Bank Penerbit Kartu")
        bank = df['payCardBank'].value_counts().reset_index()
        bank.columns = ['Bank', 'Jumlah']
        fig3 = px.bar(bank, x='Bank', y='Jumlah', title='Popularitas Bank',
                      color='Bank', template=DARK_TEMPLATE,
                      color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 10 Halte Tap-In Tersibuk")
    top_stops = df['tapInStopsName'].value_counts().head(10).reset_index()
    top_stops.columns = ['Nama Halte', 'Jumlah Penumpang']
    fig4 = px.bar(top_stops, x='Jumlah Penumpang', y='Nama Halte', orientation='h',
                  title='Halte dengan Tap-In Terbanyak', template=DARK_TEMPLATE,
                  color_discrete_sequence=COLOR_PALETTE)
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

elif menu == "Business Insights":
    st.header("Business Insights")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Durasi Perjalanan",
        "💳 Metode Pembayaran",
        "📅 Hari Kerja vs Akhir Pekan",
        "🛤️ Koridor Tersibuk",
        "📊 Distribusi Usia & Durasi"
    ])

    # --- TAB 1: Durasi Perjalanan ---
    with tab1:
        st.subheader("Rata-rata Waktu Perjalanan Antar Halte")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">📌 Mengapa insight ini penting?</div>
            Mengetahui rata-rata durasi perjalanan di rute tertentu membantu Transjakarta dalam
            mengoptimalkan jadwal armada, mengestimasi waktu tempuh untuk penumpang, dan
            mengidentifikasi rute yang perlu peningkatan efisiensi.
        </div>
        """, unsafe_allow_html=True)

        for r in ROUTES:
            mask = (
                (df['tapInStopsName'] == r['tap_in']) &
                (df['tapOutStopsName'] == r['tap_out']) &
                (df['trip_duration_mins'] > 0) &
                (df['trip_duration_mins'] < 300)
            )
            subset = df.loc[mask, 'trip_duration_mins'].dropna()

            if subset.empty:
                st.warning(f"Tidak ada data untuk rute **{r['label']}**")
                continue

            avg_val = subset.mean()
            med_val = subset.median()
            count_val = len(subset)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Rata-rata ({r['label']})", f"{avg_val:.1f} menit")
            c2.metric("Median", f"{med_val:.1f} menit")
            c3.metric("Jumlah Sampel", f"{count_val:,}")

            fig = px.histogram(subset, nbins=20, title=f"{r['label']}",
                               labels={'value': 'Durasi (menit)', 'count': 'Frekuensi'},
                               template=DARK_TEMPLATE, color_discrete_sequence=COLOR_PALETTE)
            fig.add_vline(x=avg_val, line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Rata-rata: {avg_val:.1f}")
            fig.add_vline(x=med_val, line_dash="dot", line_color="#00E5FF",
                          annotation_text=f"Median: {med_val:.1f}")
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: Metode Pembayaran per Generasi ---
    with tab2:
        st.subheader("Metode Pembayaran Favorit per Generasi")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">📌 Mengapa insight ini penting?</div>
            Memahami preferensi pembayaran setiap generasi membantu dalam strategi
            kemitraan dengan penyedia layanan pembayaran dan kampanye promosi yang
            lebih tertarget.
        </div>
        """, unsafe_allow_html=True)

        df_gen = df[['payCardBirthDate', 'payCardBank']].dropna().copy()
        df_gen['payCardBirthDate'] = pd.to_numeric(df_gen['payCardBirthDate'], errors='coerce')
        df_gen['payCardBank'] = df_gen['payCardBank'].str.strip().str.lower()
        df_gen = df_gen.dropna(subset=['payCardBirthDate'])
        df_gen['generation'] = df_gen['payCardBirthDate'].apply(assign_gen)
        df_gen = df_gen.dropna(subset=['generation'])

        gen_data = df_gen.groupby(['generation', 'payCardBank']).size().reset_index(name='count')

        for gen_name in GEN_MAP.keys():
            sub = gen_data[gen_data['generation'] == gen_name].sort_values('count', ascending=False)
            if sub.empty:
                continue
            fig = px.bar(sub, x='count', y='payCardBank', orientation='h',
                         title=gen_name, labels={'count': 'Jumlah Transaksi', 'payCardBank': 'Bank'},
                         template=DARK_TEMPLATE, color='payCardBank',
                         color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: Hari Kerja vs Akhir Pekan (Insight Tambahan) ---
    with tab3:
        st.subheader("Perbandingan Penumpang: Hari Kerja vs Akhir Pekan")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">📌 Insight tambahan</div>
            Analisis ini membantu mengidentifikasi perbedaan pola penggunaan Transjakarta
            antara hari kerja dan akhir pekan, berguna untuk perencanaan armada dan jadwal.
        </div>
        """, unsafe_allow_html=True)

        df_day = df.copy()
        df_day['TipeHari'] = df_day['tapInDayOfWeek'].apply(
            lambda x: 'Akhir Pekan' if x >= 5 else 'Hari Kerja'
        )

        day_comp = df_day.groupby(['TipeHari', 'tapInHour']).size().reset_index(name='Jumlah')
        fig_day = px.line(day_comp, x='tapInHour', y='Jumlah', color='TipeHari',
                          markers=True, title='Pola Penumpang per Jam: Hari Kerja vs Akhir Pekan',
                          labels={'tapInHour': 'Jam', 'Jumlah': 'Jumlah Penumpang'},
                          template=DARK_TEMPLATE, color_discrete_sequence=['#6C63FF', '#FF6584'])
        st.plotly_chart(fig_day, use_container_width=True)

        wd = df_day[df_day['TipeHari'] == 'Hari Kerja']
        we = df_day[df_day['TipeHari'] == 'Akhir Pekan']
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Hari Kerja", f"{len(wd):,}")
        m2.metric("Total Akhir Pekan", f"{len(we):,}")
        ratio = len(wd) / max(len(we), 1)
        m3.metric("Rasio (Kerja/Pekan)", f"{ratio:.2f}x")

    # --- TAB 4: Koridor Tersibuk (Insight Tambahan) ---
    with tab4:
        st.subheader("Top 10 Koridor Terpopuler")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">📌 Insight tambahan</div>
            Mengidentifikasi koridor tersibuk membantu prioritisasi alokasi armada
            dan perencanaan infrastruktur.
        </div>
        """, unsafe_allow_html=True)

        corr = df['corridorName'].dropna().value_counts().head(10).reset_index()
        corr.columns = ['Koridor', 'Jumlah']
        fig_corr = px.bar(corr, x='Jumlah', y='Koridor', orientation='h',
                          title='10 Koridor dengan Penumpang Terbanyak',
                          template=DARK_TEMPLATE, color_discrete_sequence=COLOR_PALETTE)
        fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Pendapatan per Koridor (Top 10)")
        rev = df.groupby('corridorName')['payAmount'].sum().dropna().sort_values(ascending=False).head(10).reset_index()
        rev.columns = ['Koridor', 'Total Pendapatan']
        fig_rev = px.bar(rev, x='Total Pendapatan', y='Koridor', orientation='h',
                         title='Top 10 Koridor berdasarkan Pendapatan',
                         template=DARK_TEMPLATE, color_discrete_sequence=['#43E97B'])
        fig_rev.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_rev, use_container_width=True)

    # --- TAB 5: Distribusi Usia & Durasi (Insight Tambahan) ---
    with tab5:
        st.subheader("Distribusi Usia Penumpang")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">📌 Insight tambahan</div>
            Memahami distribusi usia penumpang membantu dalam merancang layanan dan
            fasilitas yang sesuai dengan demografi pengguna utama.
        </div>
        """, unsafe_allow_html=True)

        age_data = df['age'].dropna()
        age_data = age_data[(age_data > 5) & (age_data < 100)]
        fig_age = px.histogram(age_data, nbins=30, title='Distribusi Usia Penumpang',
                               labels={'value': 'Usia', 'count': 'Frekuensi'},
                               template=DARK_TEMPLATE, color_discrete_sequence=['#6C63FF'])
        fig_age.add_vline(x=age_data.mean(), line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Rata-rata: {age_data.mean():.1f}")
        st.plotly_chart(fig_age, use_container_width=True)

        st.subheader("Distribusi Durasi Perjalanan")
        dur_data = df['trip_duration_mins'].dropna()
        dur_data = dur_data[(dur_data > 0) & (dur_data < 300)]
        fig_dur = px.histogram(dur_data, nbins=40, title='Distribusi Durasi Perjalanan (menit)',
                               labels={'value': 'Durasi (menit)', 'count': 'Frekuensi'},
                               template=DARK_TEMPLATE, color_discrete_sequence=['#FF6584'])
        fig_dur.add_vline(x=dur_data.mean(), line_dash="dash", line_color="#FFD700",
                          annotation_text=f"Rata-rata: {dur_data.mean():.1f}")
        fig_dur.add_vline(x=dur_data.median(), line_dash="dot", line_color="#00E5FF",
                          annotation_text=f"Median: {dur_data.median():.1f}")
        st.plotly_chart(fig_dur, use_container_width=True)

elif menu == "Prediksi Kepadatan Halte":
    st.header("Prediksi Kepadatan Penumpang di Halte")
    st.write("Gunakan Machine Learning (Linear Regression) untuk memprediksi jumlah penumpang yang melakukan Tap-In di halte tertentu pada hari dan jam yang spesifik.")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

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

        st.success(f"Diperkirakan ada **{int(round(prediction))} penumpang** Tap-In di Halte **{selected_stop}** pada hari **{selected_day}** jam **{selected_hour}:00**.")

        st.subheader("📈 Konteks Historis")
        historical = df[(df['tapInStopsName'] == selected_stop) & (df['tapInDayOfWeek'] == day_mapping[selected_day])]
        if not historical.empty:
            hist_agg = historical.groupby('tapInHour').size().reset_index(name='Jumlah')
            fig_hist = px.bar(hist_agg, x='tapInHour', y='Jumlah',
                              title=f'Historis Kepadatan Halte {selected_stop} pada Hari {selected_day}',
                              template=DARK_TEMPLATE, color_discrete_sequence=COLOR_PALETTE)
            fig_hist.add_vline(x=selected_hour, line_width=3, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Tidak ada data historis yang cukup untuk menampilkan grafik konteks.")
