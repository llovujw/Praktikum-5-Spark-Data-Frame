# Praktikum-5-Spark-Data-Frame

## BIG-DATA-PRAKTIKUM-5

### Langkah-langkah dan Code

```bash
# ========== LANGKAH 1: INSTALASI & INISIALISASI ==========

# Instalasi PySpark (jika belum)
!pip install pyspark findspark -q

# Inisialisasi SparkSession
import findspark
findspark.init()

from pyspark.sql import SparkSession

# Membuat Spark Session
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PraktikumStatistikDeskriptif") \
    .getOrCreate()

# ========== LANGKAH 2: MEMUAT DATASET ==========

import seaborn as sns

# Muat dataset 'diamonds' dari seaborn ke Pandas DataFrame
pandas_df = sns.load_dataset('diamonds')

print(f"Jumlah baris di Pandas DF: {len(pandas_df)}")
print("5 baris pertama:")
print(pandas_df.head())

# Membuat Spark DataFrame dari Pandas DataFrame
df = spark.createDataFrame(pandas_df)

print("\nSpark DataFrame:")
df.show(5)
df.printSchema()

# ========== LANGKAH 3: MENGERJAKAN LATIHAN ==========

print("\n" + "="*60)
print("MENGERJAKAN LATIHAN")
print("="*60)

# Import fungsi yang diperlukan
from pyspark.sql.functions import mean, stddev, skewness, col, min, max

# 1. Hitung statistik deskriptif untuk kolom 'carat'
print("\n=== Soal 1: Statistik Deskriptif Kolom 'carat' ===")

# a. Menggunakan .describe() untuk statistik dasar
print("\n1a. Menggunakan .describe():")
df.select("carat").describe().show()

# b. Hitung statistik spesifik
print("\n1b. Statistik spesifik:")
# Hitung mean dan stddev
mean_carat = df.select(mean("carat")).collect()[0][0]
stddev_carat = df.select(stddev("carat")).collect()[0][0]

# Hitung median dengan approxQuantile (aproksimasi)
median_carat = df.approxQuantile("carat", [0.5], 0.01)[0]

print(f"• Rata-rata (mean) carat: {mean_carat:.4f}")
print(f"• Median carat (aproksimasi): {median_carat:.4f}")
print(f"• Standar deviasi carat: {stddev_carat:.4f}")

# c. Tampilkan juga min dan max
min_carat = df.select(min("carat")).collect()[0][0]
max_carat = df.select(max("carat")).collect()[0][0]
print(f"• Minimum carat: {min_carat:.4f}")
print(f"• Maximum carat: {max_carat:.4f}")

# 2. Bandingkan rata-rata price untuk color = 'D' vs color = 'J'
print("\n" + "="*60)
print("=== Soal 2: Perbandingan Rata-rata Harga Berdasarkan Color ===")

# Filter data untuk masing-masing color
avg_price_D = df.filter(col("color") == "D").select(mean("price")).collect()[0][0]
avg_price_J = df.filter(col("color") == "J").select(mean("price")).collect()[0][0]

print(f"\nRata-rata harga untuk color 'D': ${avg_price_D:.2f}")
print(f"Rata-rata harga untuk color 'J': ${avg_price_J:.2f}")

# Bandingkan mana yang lebih mahal
print(f"\nPerbedaan harga: ${abs(avg_price_D - avg_price_J):.2f}")

if avg_price_D > avg_price_J:
    print("Kesimpulan: Color 'D' LEBIH MAHAL daripada color 'J'")
    print(f"(Selisih: ${avg_price_D - avg_price_J:.2f})")
else:
    print("Kesimpulan: Color 'J' LEBIH MAHAL daripada color 'D'")
    print(f"(Selisih: ${avg_price_J - avg_price_D:.2f})")

# 3. Buat histogram untuk kolom 'depth'
print("\n" + "="*60)
print("=== Soal 3: Analisis Distribusi Kolom 'depth' ===")

# a. Sampling data (10% seperti langkah 5.1)
sampled_depth = df.select("depth").sample(withReplacement=False, fraction=0.1, seed=42)

# b. Konversi ke Pandas DataFrame
depth_pandas_df = sampled_depth.toPandas()

print(f"\nUkuran sampel depth: {len(depth_pandas_df)} baris (dari {df.count()} total)")

# c. Hitung skewness untuk seluruh data
skewness_depth = df.select(skewness("depth")).collect()[0][0]
print(f"Nilai skewness depth (seluruh data): {skewness_depth:.3f}")

# d. Hitung statistik deskriptif untuk depth
print("\nStatistik deskriptif depth (seluruh data):")
depth_stats = df.select(
    mean("depth").alias("mean"),
    min("depth").alias("min"),
    max("depth").alias("max"),
    stddev("depth").alias("stddev")
).collect()[0]

print(f"• Mean depth: {depth_stats['mean']:.3f}")
print(f"• Min depth: {depth_stats['min']:.3f}")
print(f"• Max depth: {depth_stats['max']:.3f}")
print(f"• Stddev depth: {depth_stats['stddev']:.3f}")

# Hitung median depth
median_depth = df.approxQuantile("depth", [0.5], 0.01)[0]
print(f"• Median depth (aproksimasi): {median_depth:.3f}")

# e. Analisis distribusi
print("\nAnalisis Distribusi Depth:")
print("1. Berdasarkan nilai skewness:")
if abs(skewness_depth) < 0.5:
    print("   • Skewness mendekati 0, distribusi hampir simetris")
elif skewness_depth > 0.5:
    print(f"   • Skewness positif ({skewness_depth:.3f}), distribusi miring ke kanan")
else:
    print(f"   • Skewness negatif ({skewness_depth:.3f}), distribusi miring ke kiri")

print("\n2. Perbandingan mean vs median:")
print(f"   • Mean: {depth_stats['mean']:.3f}")
print(f"   • Median: {median_depth:.3f}")
if abs(depth_stats['mean'] - median_depth) < 0.1:
    print("   • Mean ≈ Median, distribusi simetris")
elif depth_stats['mean'] > median_depth:
    print("   • Mean > Median, distribusi miring ke kanan (positive skew)")
else:
    print("   • Mean < Median, distribusi miring ke kiri (negative skew)")

# f. Visualisasi histogram
import matplotlib.pyplot as plt
import seaborn as sns

print("\n3. Membuat visualisasi histogram...")

plt.figure(figsize=(12, 5))

# Subplot 1: Histogram dengan KDE
plt.subplot(1, 2, 1)
sns.histplot(depth_pandas_df['depth'], kde=True, bins=30, color='skyblue')
plt.title('Distribusi Depth (Sampel 10%)', fontsize=14)
plt.xlabel('Depth', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.grid(True, alpha=0.3)

# Tambahkan garis vertikal untuk mean dan median
plt.axvline(x=depth_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {depth_stats["mean"]:.2f}')
plt.axvline(x=median_depth, color='green', linestyle='--', linewidth=2, label=f'Median: {median_depth:.2f}')
plt.legend()

# Subplot 2: Box plot
plt.subplot(1, 2, 2)
sns.boxplot(y=depth_pandas_df['depth'], color='lightgreen')
plt.title('Box Plot Depth', fontsize=14)
plt.ylabel('Depth', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KESIMPULAN DISTRIBUSI DEPTH:")
print("="*60)
print("Berdasarkan analisis statistik dan visualisasi:")
print("1. Distribusi depth terlihat MENDEKATI NORMAL (bell-shaped)")
print("2. Terdapat sedikit kemiringan (skew) positif")
print("3. Tidak terlihat pola bimodal (dua puncak)")
print("4. Terdapat beberapa outliers di kedua sisi distribusi")
print("5. Rentang data cukup terkonsentrasi di sekitar mean")

# Stop Spark session untuk membersihkan sumber daya
spark.stop()
print("\nSpark session telah dihentikan.")
