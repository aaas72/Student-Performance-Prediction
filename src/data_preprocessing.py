import pandas as pd
from sklearn.impute import SimpleImputer
import os

# Dosya yolunu sabit olarak belirle
file_path = "dataset/student/student-mat.csv"

# Dosyanın varlığını kontrol et
if not os.path.exists(file_path):
    print("Hata: Dosya bulunamadı. Lütfen 'dataset/student/student-mat.csv' dosyasının doğru yerde olduğundan emin olun.")
    exit()

# CSV dosyasını oku ve hata yönetimi yap
try:
    data = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()
except Exception as e:
    print(f"Hata oluştu: {e}")
    exit()

# Eksik verileri hesapla
missing_data = data.isnull().sum()
# Eksik veri yüzdesini hesapla
missing_percentage = (missing_data / len(data)) * 100

# Kategorik sütunları belirle (metin türündeki sütunlar)
categorical_columns = data.select_dtypes(include=['object']).columns
# Sayısal sütunları belirle (sayı türündeki sütunlar)
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Kategorik sütunlardaki eksik verileri en sık geçen değerle doldur
categorical_transformer = SimpleImputer(strategy='most_frequent')
for col in categorical_columns:
    data[col] = categorical_transformer.fit_transform(data[[col]]).ravel()

# Sayısal sütunlardaki eksik verileri ortalama değerle doldur
numerical_transformer = SimpleImputer(strategy='mean')  
data[numerical_columns] = numerical_transformer.fit_transform(data[numerical_columns])

# Tekrar eden satırları kontrol et ve kaldır
duplicates = data.duplicated().sum()
if duplicates > 0:
    print(f"{duplicates} tekrar eden satır bulundu ve kaldırıldı.")
    data = data.drop_duplicates()

# Eksik veri raporunu yazdır
print("\nEksik veri raporu:")
for col in data.columns:
    if missing_data[col] > 0:
        print(f"{col}: {missing_data[col]} eksik değer, "
              f"{missing_percentage[col]:.2f}% oranında, "
              f"{'ortalama' if col in numerical_columns else 'en sık değer'} ile dolduruldu.")
    else:
        print(f"{col}: Eksik veri yok.")

# İşlendikten sonra eksik veri kontrolü yap
print("\nVeriler işlendikten sonra eksik veri kontrolü:")
print(data.isnull().sum())  # Eksik veri kalıp kalmadığını kontrol et

# İşlenmiş verileri yeni bir dosyaya kaydet
try:
    output_file = file_path.replace('.csv', '_processed_file.csv')
    data.to_csv(output_file, index=False)
    print(f"İşlenmiş veriler şu dosyaya kaydedildi: {output_file}")
except Exception as e:
    print(f"Verileri kaydederken hata oluştu: {e}")
    
    
    
    