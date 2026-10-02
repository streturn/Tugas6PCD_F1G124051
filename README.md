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

## How to Run

### 1. Clone Repository
Buka Command Prompt atau PowerShell, kemudian jalankan:
```bash
git clone https://github.com/streturn/Tugas6PCD_F1G124051.git
```

Kemudian masuk ke folder project:
```bash
cd Tugas6PCD_F1G124051
```

### 2. Install Library
Install library yang diperlukan dengan perintah:
```bash
pip install opencv-python numpy matplotlib
```

### 3. Jalankan Program
Jalankan program dengan:
```bash
python main.py
```

---

## Hasil Pengolahan

Program melakukan beberapa tahap pengolahan:
* Grayscale
* Global Thresholding
* Otsu Thresholding
* Opening
* Closing

Hasil pengolahan disimpan pada folder:
`hasil/`

---

## Metode yang Digunakan

* **Global Thresholding**  
  Global threshold digunakan untuk memisahkan foreground dan background menggunakan nilai ambang tertentu. Pada program ini digunakan nilai threshold 127.

* **Otsu Thresholding**  
  Metode Otsu menentukan nilai threshold secara otomatis berdasarkan distribusi intensitas citra.

* **Opening**  
  Opening digunakan untuk mengurangi noise kecil pada hasil thresholding.

* **Closing**  
  Closing digunakan untuk membantu menghubungkan bagian foreground yang terputus dan mengisi celah kecil.

---

## Output

Output program berupa:
* Citra hasil crop
* Citra grayscale
* Hasil Global Thresholding
* Hasil Otsu Thresholding
* Hasil Opening
* Hasil Closing
* Jumlah foreground pixel
* Status keputusan (SIGNATURE PRESENT / SIGNATURE ABSENT)
* File `rekapitulasi_hasil.csv`

---

## Dataset

Dataset yang digunakan terdiri dari beberapa citra dokumen ijazah dengan kondisi kualitas citra yang berbeda.  
Contohnya:
* High Quality
* Low Contrast
* Blurred
* High Noise
* Low Resolution
* Faded / Underexposed
* Color Shift
* JPEG Compression
* Combined Degradation

---

## Author

* **Nama:** St. Rahmy
* **NIM:** F1G124051
* **Mata Kuliah:** Pengolahan Citra Digital
* **Universitas:** Universitas Halu Oleo