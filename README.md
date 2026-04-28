# Maize Vision

Maize Vision adalah server FastAPI untuk klasifikasi penyakit daun jagung berbasis deep learning. Proyek ini memakai dua arsitektur unggulan, ConvNeXt dan MaxViT, untuk memberikan prediksi kelas penyakit dan visualisasi fokus model melalui Grad-CAM. Hasilnya: prediksi yang informatif dan mudah dipahami untuk kebutuhan riset maupun demo aplikasi.

## Nilai Utama

- Dual-architecture: pilih ConvNeXt atau MaxViT sesuai kebutuhan performa.
- Prediksi + visualisasi: keluaran mencakup skor kelas dan citra Grad-CAM.
- API sederhana: cukup kirim satu gambar lewat endpoint tunggal.
- Siap GPU/CPU: otomatis memakai CUDA bila tersedia.

## Arsitektur Model

- ConvNeXt (Tiny): modernisasi CNN klasik untuk akurasi kuat dan efisiensi tinggi.
- MaxViT (Tiny): kombinasi convolution dan windowed attention untuk pemahaman konteks yang kaya.

Keduanya diakses lewat satu endpoint sehingga Anda bisa membandingkan hasil secara cepat.

## Alur Prediksi

1. Unggah gambar daun jagung.
2. Server melakukan preprocessing dan inferensi.
3. Grad-CAM dihasilkan sebagai overlay fokus model.
4. Respons mengembalikan skor kelas dan citra hasil.

## Label Kelas

Model mengembalikan prediksi untuk label berikut:

- Blight
- Common_Rust
- Gray_Leaf_Spot
- Healthy
- Pest_Damage

## Endpoint API

### GET /

Mengembalikan pesan sederhana untuk memastikan server aktif.

### POST /predict

Form data:

- `model`: `convnext` atau `maxvit`
- `image`: file gambar (jpg, jpeg, png, webp)

Respons JSON:

- `original_image`: citra asli (base64 data URL)
- `grad_cam_image`: citra Grad-CAM (base64 data URL)
- `scores`: daftar skor dalam persen
- `class_names`: daftar label kelas

## Cara Menjalankan

Prasyarat:

- Python 3.13.12
- pip 26.0.1

Langkah cepat (Windows PowerShell):

```bash
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
fastapi run --workers 2 ./app/main.py
```

Server akan aktif di http://127.0.0.1:8000.

## Contoh Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
	-F "model=convnext" ^
	-F "image=@path\to\image.jpg"
```

Ganti `model=convnext` dengan `model=maxvit` untuk mencoba arsitektur lain.

## Catatan

- Model weight disimpan di folder `model_weights` dan dimuat saat server start.
- Gunakan gambar RGB untuk hasil terbaik (server akan mengonversi jika perlu).
