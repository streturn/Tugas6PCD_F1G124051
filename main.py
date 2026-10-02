import cv2
import numpy as np
import pandas as pd
import os
import glob


# ============================================================
# 1. KONFIGURASI
# ============================================================

INPUT_FOLDER = "citra"
OUTPUT_FOLDER = "hasil"


# ============================================================
# 2. AREA CROP TANDA TANGAN DEKAN
# ============================================================
#
# Gambar asli diputar 90 derajat clockwise terlebih dahulu.
#
# Crop menggunakan persentase supaya lebih fleksibel
# terhadap perbedaan ukuran gambar.
#
# Berdasarkan gambar ijazah yang digunakan:
# tanda tangan Dekan berada di bagian kanan bawah.
# ============================================================

X1_RATIO = 0.61
Y1_RATIO = 0.70

X2_RATIO = 0.89
Y2_RATIO = 0.82


# ============================================================
# 3. KONFIGURASI THRESHOLD
# ============================================================

# Threshold Global
GLOBAL_THRESHOLD = 145

# Threshold untuk membuat citra simulasi tanpa tanda tangan
REMOVE_SIGNATURE_THRESHOLD = 150

# Ukuran kernel morphology
KERNEL_SIZE = 3


# ============================================================
# 4. MEMBUAT FOLDER OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_FOLDER,
        "ada_ttd"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_FOLDER,
        "tanpa_ttd"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_FOLDER,
        "perbandingan"
    ),
    exist_ok=True
)


# ============================================================
# 5. ROTASI DAN CROP
# ============================================================

def rotate_and_crop(image):

    # --------------------------------------------------------
    # Putar gambar 90 derajat searah jarum jam
    # --------------------------------------------------------

    rotated = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

    # --------------------------------------------------------
    # Ambil ukuran gambar
    # --------------------------------------------------------

    h, w = rotated.shape[:2]

    # --------------------------------------------------------
    # Ubah koordinat persentase menjadi pixel
    # --------------------------------------------------------

    x1 = int(w * X1_RATIO)
    y1 = int(h * Y1_RATIO)

    x2 = int(w * X2_RATIO)
    y2 = int(h * Y2_RATIO)

    # --------------------------------------------------------
    # Crop area tanda tangan Dekan
    # --------------------------------------------------------

    crop = rotated[
        y1:y2,
        x1:x2
    ]

    return rotated, crop


# ============================================================
# 6. GRAYSCALE
# ============================================================

def convert_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# ============================================================
# 7. GLOBAL THRESHOLD
# ============================================================

def global_threshold(gray):

    _, binary = cv2.threshold(
        gray,
        GLOBAL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    return binary


# ============================================================
# 8. OTSU THRESHOLD
# ============================================================

def otsu_threshold(gray):

    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary, threshold_value


# ============================================================
# 9. MORPHOLOGY
# ============================================================

def apply_morphology(binary):

    # Kernel berbentuk ellipse
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            KERNEL_SIZE,
            KERNEL_SIZE
        )
    )

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------
    # Menghilangkan noise kecil
    # --------------------------------------------------------

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # CLOSING
    # --------------------------------------------------------
    # Menutup celah kecil pada objek
    # --------------------------------------------------------

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closing


# ============================================================
# 10. HITUNG FOREGROUND PIXEL
# ============================================================

def calculate_foreground(binary):

    # Pixel putih dianggap sebagai foreground
    foreground_pixels = np.count_nonzero(
        binary
    )

    # Total pixel
    total_pixels = binary.size

    # Persentase foreground
    foreground_ratio = (
        foreground_pixels /
        total_pixels
    ) * 100

    return (
        foreground_pixels,
        foreground_ratio
    )


# ============================================================
# 11. MEMBUAT CITRA TANPA TANDA TANGAN
# ============================================================

def create_without_signature(crop):

    # Ubah crop ke grayscale
    gray = convert_grayscale(
        crop
    )

    # --------------------------------------------------------
    # Membuat mask area gelap
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        gray,
        REMOVE_SIGNATURE_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    # --------------------------------------------------------
    # Hilangkan noise kecil pada mask
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Perbesar mask sedikit
    # supaya bagian tanda tangan ikut terhapus
    # --------------------------------------------------------

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Hilangkan tanda tangan dengan inpainting
    # --------------------------------------------------------

    without_signature = cv2.inpaint(
        crop,
        mask,
        5,
        cv2.INPAINT_TELEA
    )

    return without_signature


# ============================================================
# 12. SIMPAN PERBANDINGAN
# ============================================================

def save_comparison(
    gray,
    global_result,
    otsu_result,
    output_path
):

    # --------------------------------------------------------
    # Ubah grayscale menjadi BGR
    # agar dapat diberi tulisan
    # --------------------------------------------------------

    gray_image = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    global_image = cv2.cvtColor(
        global_result,
        cv2.COLOR_GRAY2BGR
    )

    otsu_image = cv2.cvtColor(
        otsu_result,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Tambahkan judul
    # --------------------------------------------------------

    cv2.putText(
        gray_image,
        "GRAYSCALE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        global_image,
        "GLOBAL THRESHOLD",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        otsu_image,
        "OTSU THRESHOLD",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # Gabungkan secara horizontal
    # --------------------------------------------------------

    comparison = cv2.hconcat(
        [
            gray_image,
            global_image,
            otsu_image
        ]
    )

    # --------------------------------------------------------
    # Simpan
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        comparison
    )


# ============================================================
# 13. PROSES SATU CITRA
# ============================================================

def process_image(
    crop,
    output_prefix,
    comparison_name
):

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = convert_grayscale(
        crop
    )

    # --------------------------------------------------------
    # GLOBAL THRESHOLD
    # --------------------------------------------------------

    global_binary = global_threshold(
        gray
    )

    global_morph = apply_morphology(
        global_binary
    )

    global_pixels, global_ratio = \
        calculate_foreground(
            global_morph
        )

    # --------------------------------------------------------
    # OTSU THRESHOLD
    # --------------------------------------------------------

    otsu_binary, otsu_value = \
        otsu_threshold(
            gray
        )

    otsu_morph = apply_morphology(
        otsu_binary
    )

    otsu_pixels, otsu_ratio = \
        calculate_foreground(
            otsu_morph
        )

    # ========================================================
    # SIMPAN HASIL INDIVIDUAL
    # ========================================================

    cv2.imwrite(
        output_prefix + "_crop.jpg",
        crop
    )

    cv2.imwrite(
        output_prefix + "_grayscale.jpg",
        gray
    )

    cv2.imwrite(
        output_prefix + "_global.jpg",
        global_morph
    )

    cv2.imwrite(
        output_prefix + "_otsu.jpg",
        otsu_morph
    )

    # ========================================================
    # SIMPAN PERBANDINGAN
    # ========================================================

    comparison_path = os.path.join(
        OUTPUT_FOLDER,
        "perbandingan",
        comparison_name
    )

    save_comparison(
        gray,
        global_morph,
        otsu_morph,
        comparison_path
    )

    # ========================================================
    # RETURN HASIL
    # ========================================================

    return {

        "global_pixels":
            global_pixels,

        "global_ratio":
            global_ratio,

        "otsu_pixels":
            otsu_pixels,

        "otsu_ratio":
            otsu_ratio,

        "otsu_threshold":
            otsu_value
    }


# ============================================================
# 14. MENCARI SEMUA GAMBAR
# ============================================================

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png"
]:

    image_files.extend(
        glob.glob(
            os.path.join(
                INPUT_FOLDER,
                extension
            )
        )
    )


image_files = sorted(
    image_files
)


# ============================================================
# 15. CEK GAMBAR
# ============================================================

if len(image_files) == 0:

    print()
    print(
        "ERROR: Tidak ada gambar ditemukan!"
    )

    print(
        "Masukkan gambar ke folder:",
        INPUT_FOLDER
    )

    exit()


print()
print("=" * 80)
print("SIGNATURE PRESENCE DETECTION")
print("=" * 80)

print(
    f"Jumlah gambar ditemukan: "
    f"{len(image_files)}"
)


# ============================================================
# 16. MENYIMPAN HASIL
# ============================================================

results = []


# ============================================================
# 17. PROSES CITRA ASLI
# ============================================================
#
# Semua gambar asli dianggap memiliki tanda tangan.
#
# ============================================================

print()
print("=" * 80)
print("TAHAP 1 : CITRA ASLI / ADA TANDA TANGAN")
print("=" * 80)


for file in image_files:

    filename = os.path.basename(
        file
    )

    image = cv2.imread(
        file
    )

    if image is None:

        print(
            f"Gagal membaca: {filename}"
        )

        continue

    # --------------------------------------------------------
    # ROTASI + CROP
    # --------------------------------------------------------

    rotated, crop = \
        rotate_and_crop(
            image
        )

    # --------------------------------------------------------
    # Nama file
    # --------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]

    # --------------------------------------------------------
    # Prefix output
    # --------------------------------------------------------

    output_prefix = os.path.join(
        OUTPUT_FOLDER,
        "ada_ttd",
        base_name
    )

    # --------------------------------------------------------
    # Nama perbandingan
    # --------------------------------------------------------

    comparison_name = (
        base_name +
        "_ADA_TTD_comparison.jpg"
    )

    # --------------------------------------------------------
    # Proses
    # --------------------------------------------------------

    data = process_image(
        crop,
        output_prefix,
        comparison_name
    )

    # --------------------------------------------------------
    # SIMPAN HASIL GLOBAL
    # --------------------------------------------------------

    results.append({

        "Citra":
            filename,

        "Kondisi":
            "ADA TANDA TANGAN",

        "Metode":
            "Global",

        "Threshold":
            GLOBAL_THRESHOLD,

        "Foreground_Pixel":
            data["global_pixels"],

        "Foreground_Ratio":
            data["global_ratio"],

        "Prediksi":
            ""
    })

    # --------------------------------------------------------
    # SIMPAN HASIL OTSU
    # --------------------------------------------------------

    results.append({

        "Citra":
            filename,

        "Kondisi":
            "ADA TANDA TANGAN",

        "Metode":
            "Otsu",

        "Threshold":
            data["otsu_threshold"],

        "Foreground_Pixel":
            data["otsu_pixels"],

        "Foreground_Ratio":
            data["otsu_ratio"],

        "Prediksi":
            ""
    })

    print(
        f"{filename:<45}"
        f"Global = "
        f"{data['global_ratio']:.2f}% | "
        f"Otsu = "
        f"{data['otsu_ratio']:.2f}%"
    )


# ============================================================
# 18. MEMBUAT DATA SIMULASI TANPA TANDA TANGAN
# ============================================================

print()
print("=" * 80)
print("TAHAP 2 : DATA SIMULASI / TANPA TANDA TANGAN")
print("=" * 80)


for file in image_files:

    filename = os.path.basename(
        file
    )

    image = cv2.imread(
        file
    )

    if image is None:
        continue

    # --------------------------------------------------------
    # ROTASI + CROP
    # --------------------------------------------------------

    rotated, crop = \
        rotate_and_crop(
            image
        )

    # --------------------------------------------------------
    # HAPUS TANDA TANGAN
    # --------------------------------------------------------

    without_signature = \
        create_without_signature(
            crop
        )

    # --------------------------------------------------------
    # Nama file
    # --------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]

    # --------------------------------------------------------
    # Prefix output
    # --------------------------------------------------------

    output_prefix = os.path.join(
        OUTPUT_FOLDER,
        "tanpa_ttd",
        base_name
    )

    # --------------------------------------------------------
    # Nama perbandingan
    # --------------------------------------------------------

    comparison_name = (
        base_name +
        "_TANPA_TTD_comparison.jpg"
    )

    # --------------------------------------------------------
    # Proses
    # --------------------------------------------------------

    data = process_image(
        without_signature,
        output_prefix,
        comparison_name
    )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    results.append({

        "Citra":
            filename + "_TANPA_TTD",

        "Kondisi":
            "TIDAK ADA TANDA TANGAN",

        "Metode":
            "Global",

        "Threshold":
            GLOBAL_THRESHOLD,

        "Foreground_Pixel":
            data["global_pixels"],

        "Foreground_Ratio":
            data["global_ratio"],

        "Prediksi":
            ""
    })

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    results.append({

        "Citra":
            filename + "_TANPA_TTD",

        "Kondisi":
            "TIDAK ADA TANDA TANGAN",

        "Metode":
            "Otsu",

        "Threshold":
            data["otsu_threshold"],

        "Foreground_Pixel":
            data["otsu_pixels"],

        "Foreground_Ratio":
            data["otsu_ratio"],

        "Prediksi":
            ""
    })

    print(
        f"{filename:<45}"
        f"Global = "
        f"{data['global_ratio']:.2f}% | "
        f"Otsu = "
        f"{data['otsu_ratio']:.2f}%"
    )


# ============================================================
# 19. BUAT DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# 20. MENENTUKAN RULE KLASIFIKASI
# ============================================================

global_data = df[
    df["Metode"] == "Global"
].copy()


# ------------------------------------------------------------
# Data ADA TTD
# ------------------------------------------------------------

present_data = global_data[
    global_data["Kondisi"]
    == "ADA TANDA TANGAN"
]


# ------------------------------------------------------------
# Data TANPA TTD
# ------------------------------------------------------------

absent_data = global_data[
    global_data["Kondisi"]
    == "TIDAK ADA TANDA TANGAN"
]


# ------------------------------------------------------------
# Nilai minimum foreground pada data ADA TTD
# ------------------------------------------------------------

min_present = present_data[
    "Foreground_Ratio"
].min()


# ------------------------------------------------------------
# Nilai maksimum foreground pada data TANPA TTD
# ------------------------------------------------------------

max_absent = absent_data[
    "Foreground_Ratio"
].max()


# ------------------------------------------------------------
# Threshold keputusan
# ------------------------------------------------------------

RULE_THRESHOLD = (
    min_present +
    max_absent
) / 2


print()
print("=" * 80)
print("ATURAN KLASIFIKASI")
print("=" * 80)

print(
    f"Foreground minimum ADA TTD   : "
    f"{min_present:.2f}%"
)

print(
    f"Foreground maksimum TANPA TTD: "
    f"{max_absent:.2f}%"
)

print(
    f"Threshold keputusan          : "
    f"{RULE_THRESHOLD:.2f}%"
)


# ============================================================
# 21. FUNGSI KLASIFIKASI
# ============================================================

def classify_signature(
    foreground_ratio
):

    if foreground_ratio >= \
            RULE_THRESHOLD:

        return "SIGNATURE PRESENT"

    else:

        return "SIGNATURE ABSENT"


# ============================================================
# 22. PREDIKSI GLOBAL
# ============================================================

df.loc[
    df["Metode"] == "Global",
    "Prediksi"
] = df.loc[
    df["Metode"] == "Global",
    "Foreground_Ratio"
].apply(
    classify_signature
)


# Otsu hanya digunakan untuk membandingkan
# hasil thresholding.
df.loc[
    df["Metode"] == "Otsu",
    "Prediksi"
] = "PERBANDINGAN"


# ============================================================
# 23. EVALUASI
# ============================================================

evaluation = df[
    df["Metode"] == "Global"
].copy()


# ------------------------------------------------------------
# Tentukan benar atau salah
# ------------------------------------------------------------

evaluation["Benar"] = (

    (
        (
            evaluation["Kondisi"]
            == "ADA TANDA TANGAN"
        )
        &
        (
            evaluation["Prediksi"]
            == "SIGNATURE PRESENT"
        )
    )

    |

    (
        (
            evaluation["Kondisi"]
            == "TIDAK ADA TANDA TANGAN"
        )
        &
        (
            evaluation["Prediksi"]
            == "SIGNATURE ABSENT"
        )
    )

)


# ============================================================
# 24. HITUNG AKURASI
# ============================================================

accuracy = (
    evaluation["Benar"].mean()
) * 100


# ============================================================
# 25. SIMPAN HASIL CSV
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_threshold.csv"
    ),
    index=False
)


evaluation.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_pengujian.csv"
    ),
    index=False
)


# ============================================================
# 26. TAMPILKAN HASIL
# ============================================================

print()
print("=" * 110)
print("HASIL PENGUJIAN SISTEM")
print("=" * 110)

print(
    f"{'Citra':<45}"
    f"{'Kondisi':<25}"
    f"{'Foreground':<15}"
    f"{'Prediksi'}"
)

print(
    "-" * 110
)


for _, row in evaluation.iterrows():

    print(
        f"{row['Citra']:<45}"
        f"{row['Kondisi']:<25}"
        f"{row['Foreground_Ratio']:.2f}%"
        f"{'':<8}"
        f"{row['Prediksi']}"
    )


# ============================================================
# 27. HASIL AKHIR
# ============================================================

print()
print("=" * 80)

print(
    f"AKURASI SISTEM : "
    f"{accuracy:.2f}%"
)

print("=" * 80)

print()
print(
    "Folder hasil:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)

print()
print(
    "File CSV:"
)

print(
    "- hasil/hasil_threshold.csv"
)

print(
    "- hasil/hasil_pengujian.csv"
)

print()
print(
    "Proses selesai."
)