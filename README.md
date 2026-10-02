# Signature Presence Detection

Mini Project Pengolahan Citra Digital untuk mendeteksi keberadaan tanda tangan Dekan pada citra ijazah.

## Deskripsi

Program ini digunakan untuk mendeteksi apakah suatu area tanda tangan pada citra ijazah memiliki tanda tangan atau tidak.

Tahapan utama yang digunakan dalam program adalah:

1. Crop area tanda tangan Dekan
2. Konversi citra RGB menjadi grayscale
3. Global Thresholding
4. Otsu Thresholding
5. Morphological Opening
6. Morphological Closing
7. Perhitungan jumlah foreground pixel
8. Penentuan kondisi tanda tangan
9. Pengujian citra dengan tanda tangan dan tanpa tanda tangan

Karena seluruh citra ijazah asli yang digunakan memiliki tanda tangan, citra tanpa tanda tangan dibuat sebagai data simulasi dengan menghilangkan area tanda tangan menggunakan teknik inpainting.

---

## Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- Pandas

---

## Struktur Folder

```text
signature-detection/
│
├── main.py
├── README.md
├── requirements.txt
│
├── citra/
│   ├── 01_HighQuality_Enhanced.jpg
│   ├── 02_LowContrast.jpg
│   ├── 03_Blurred.jpg
│   ├── 04_HighNoise.jpg
│   ├── 05_LowResolution_Upsampled.jpg
│   ├── 06_Faded_Underexposed.jpg
│   ├── 07_ColorShift_WarmTint.jpg
│   ├── 08_JPEGCompression_Artifacts.jpg
│   └── 09_CombinedDegradation.jpg
│
└── hasil/
    ├── ada_ttd/
    ├── tanpa_ttd/
    ├── perbandingan/
    ├── hasil_threshold.csv
    └── hasil_pengujian.csv