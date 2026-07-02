---
title: Maize Vision Server
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Maize Vision

Maize Vision adalah server FastAPI untuk inferensi citra daun jagung berbasis ConvNeXt dan model binary scikit-learn. Sistem menghasilkan prediksi multi-kelas penyakit, visualisasi fokus model melalui Grad-CAM, serta prediksi binary untuk mendeteksi apakah gambar termasuk daun jagung atau bukan.

## Nilai Utama

- Satu alur inferensi: ConvNeXt untuk klasifikasi penyakit multi-kelas.
- Prediksi + visualisasi: keluaran mencakup skor kelas dan citra Grad-CAM.
- Verifikasi binary tambahan: model scikit-learn memprediksi `Bukan Daun Jagung` vs `Daun Jagung`.
- API sederhana: cukup kirim satu gambar lewat endpoint tunggal.
- Siap GPU/CPU: otomatis memakai CUDA bila tersedia.

## Arsitektur Model

- ConvNeXt (Tiny): model utama untuk klasifikasi 5 kelas penyakit daun jagung.
- Binary classifier (scikit-learn): model `SVC` terkalibrasi probabilitas yang berjalan di atas fitur ConvNeXt (`avgpool`).

MaxViT tidak digunakan pada alur inferensi aktif.

Pipeline binary classifier:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
	("scaler", StandardScaler()),
	(
		"svm",
		CalibratedClassifierCV(
			estimator=SVC(),
			method="sigmoid",
			ensemble=False,
			cv=5,
		),
	),
])
```

## Alur Prediksi

1. Unggah gambar daun jagung.
2. Server melakukan preprocessing dan inferensi.
3. Fitur `avgpool` ConvNeXt dipakai untuk inferensi binary classifier.
4. Grad-CAM dihasilkan sebagai overlay fokus model.
5. Respons mengembalikan skor multi-kelas, skor binary, dan citra hasil.

## Label Kelas

Model mengembalikan prediksi untuk label berikut:

- Hawar Daun
- Karat Daun
- Bercak Daun
- Sehat
- Kerusakan Hama

Label binary tambahan:

- Bukan Daun Jagung
- Daun Jagung

## Endpoint API

### GET /

Mengembalikan pesan sederhana untuk memastikan server aktif.

### POST /predict

Form data:

- `image`: file gambar (jpg, jpeg, png, webp)

Respons JSON:

- `original_image`: citra asli (base64 data URL)
- `grad_cam_image`: citra Grad-CAM (base64 data URL)
- `scores`: daftar skor dalam persen
- `class_names`: daftar label kelas
- `binary_scores`: daftar skor binary dalam persen
- `binary_class_names`: daftar label binary

## Cara Menjalankan

Prasyarat:

- Python 3.13.12
- pip 26.0.1

Langkah cepat (Windows PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
fastapi run --workers 2 ./app/main.py
```

Server akan aktif di http://127.0.0.1:8000.

## Contoh Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
	-F "image=@path\to\image.jpg"
```

## Catatan

- Model weight disimpan di folder `model_weights` dan dimuat saat server start.
- File model yang digunakan saat ini: `convnext_weights.pth` dan `binary.joblib`.
- Gunakan gambar RGB untuk hasil terbaik (server akan mengonversi jika perlu).
