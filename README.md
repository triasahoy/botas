Oke, Trias — ini **README.md final** yang sudah lengkap, rapi, dan profesional.  
Aku gabungkan semua yang kemarin kita susun, plus aku tambahkan **badge** di bagian atas untuk memberi kesan modern dan kredibel.

---

```markdown
# 📈 DCA BTC/IDR Tokocrypto — Modern Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dashboard interaktif berbasis **Streamlit** untuk menganalisis strategi **Dollar Cost Averaging (DCA)** pada pasangan **BTC/IDR** menggunakan data dari **Tokocrypto** (via CCXT).  
Mendukung multi‑slippage, perhitungan biaya transaksi, benchmark Buy & Hold, ekspor hasil ke ZIP, dan tampilan grafik equity curve yang modern.

---

## ✨ Fitur Utama
- **Multi‑slippage backtest**: uji performa DCA dengan berbagai asumsi slippage (bps).
- **Benchmark Buy & Hold**: bandingkan hasil DCA dengan strategi beli‑tahan.
- **Ekspor ZIP**: simpan hasil backtest (summary, breakdown, equity curve) dalam satu file ZIP.
- **Format Indonesia**: angka dan mata uang otomatis diformat sesuai lokal.
- **Auto‑refresh harga**: update harga BTC/IDR setiap 60 detik.
- **UI modern**: grafik interaktif dengan Plotly.

---

## 📦 Instalasi Lokal

1. **Clone repo**
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   Pastikan `requirements.txt` ada di root repo:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables** (API Key Tokocrypto)
   ```bash
   export TOKO_API_KEY="API_KEY_ANDA"
   export TOKO_SECRET="SECRET_ANDA"
   ```
   > Di Windows (PowerShell):
   ```powershell
   setx TOKO_API_KEY "API_KEY_ANDA"
   setx TOKO_SECRET "SECRET_ANDA"
   ```

5. **Jalankan aplikasi**
   ```bash
   streamlit run app_dca_tokocrypto_id_modern_plus.py
   ```

---

## ☁️ Deploy ke Streamlit Cloud

1. Push repo ini ke GitHub.
2. Buka [Streamlit Cloud](https://share.streamlit.io/), login dengan akun GitHub.
3. Pilih **New app** → pilih repo dan branch.
4. Pastikan file utama:  
   ```
   app_dca_tokocrypto_id_modern_plus.py
   ```
5. Tambahkan **Secrets** di menu Settings:
   ```toml
   TOKO_API_KEY = "API_KEY_ANDA"
   TOKO_SECRET = "SECRET_ANDA"
   ```
6. Deploy dan nikmati dashboard DCA Anda secara online.

---

## 📄 requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
ccxt>=4.1.0
python-dateutil>=2.8.2
pytz>=2023.3
Babel>=2.12.1
streamlit-autorefresh>=0.0.1
```

---

## 📄 Lisensi
Proyek ini dirilis di bawah lisensi MIT — silakan digunakan, dimodifikasi, dan dibagikan.

---

## 🙌 Kontribusi
Pull request dan masukan sangat diterima.  
Silakan buka *issue* untuk bug report atau ide fitur baru.

---

**Dibuat oleh:** Trias  
**Teknologi:** Python, Streamlit, CCXT, Plotly, Pandas
```

---
