# Telco Customer Churn Proje Raporu (Kisa)

## 1. Proje Amaci
Bu projenin amaci, musteri kaybini (churn) erken tespit etmek ve veri odakli aksiyonlarla kaybi azaltmaktir.

## 2. Problem Tanimi
Telekom sektorunde musteri kaybi gelir azalmasina ve yeni musteri kazanma maliyetlerinin artmasina neden olur. Bu nedenle riskli musteri gruplarinin erken belirlenmesi kritik oneme sahiptir.

## 3. Veri Seti
Kullanilan veri seti: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Gozlem sayisi: 7043
- Degisken sayisi: 21
- Hedef degisken: Churn (Yes/No)
- Veri temizleme: TotalCharges degiskenindeki bos degerler sayisala cevrilip modelleme asamasinda imputasyon ile tamamlandi.

## 4. Yontem
Projede Python tabanli bir Streamlit arayuzu gelistirildi.
- Veri analizi: churn dagilimi, sozlesme tipi, tenure ve aylik ucret iliskileri
- Modelleme: Random Forest siniflandirma modeli
- Degerlendirme metrikleri: Accuracy, ROC-AUC, Confusion Matrix, Classification Report
- Cozum katmani: Yuksek riskli musterileri belirleme, segment bazli aksiyon onerileri ve kampanya etki simulasyonu

## 5. Temel Bulgular
- Veri setinde churn orani yaklasik %26.5 seviyesindedir.
- Month-to-month sozlesme tipinde churn riski daha yuksek gorunmektedir.
- Model performansi:
  - Accuracy: 0.7814
  - ROC-AUC: 0.8224
Bu sonuc, modelin churn tahmini icin kullanilabilir bir performans sundugunu gostermektedir.

## 6. Onerilen Cozum
Model skoru ile yuksek riskli musteriler belirlenir ve su aksiyonlar uygulanir:
- Uzun donem kontrata gecis tesviki
- Paket/fatura optimizasyonu
- Teknik destek ve guvenlik paketi kampanyalari
- Yeni musteri (dusuk tenure) icin sadakat teklifleri

Simulasyon ekraninda kampanya maliyeti ve beklenen basari orani girilerek net finansal etki hesaplanir.

## 7. Sonuc
Bu proje, sadece churn tahmini yapmakla kalmayip is problemi icin uygulanabilir cozum de uretmektedir. Gelistirilen sistem ile hem riskli musteriler tespit edilmekte hem de retention aksiyonlarinin beklenen etkisi olculebilmektedir.
