MemeFace AR
MediaPipe ve OpenCV kütüphaneleri kullanılarak geliştirilmiş, gerçek zamanlı el ve yüz takibi yapan bir artırılmış gerçeklik (AR) uygulamasıdır. Belirlenen el hareketlerine göre kullanıcı yüzüne dinamik filtreler uygular.

İşlem adımları
Görüntü İşleme: OpenCV kütüphanesi aracılığıyla kamera akışı saniyelik kareler halinde yakalanır ve işlenir.

Yapay Zeka Analizi: MediaPipe kütüphanesi kullanılarak el üzerindeki 21 farklı eklem noktası (landmarks) ve yüz çerçevesi gerçek zamanlı olarak tespit edilir.

Filtre Uygulama: İşaret parmağının dikey koordinatı analiz edilerek (landmark 8 vs landmark 6), belirlenen görselin yüz koordinatlarına (Bounding Box) bindirilmesi sağlanır.

Stabilite: Sistem, kütüphane çakışmalarını önlemek amacıyla belirli NumPy ve OpenCV sürümleriyle optimize edilmiştir.

Kurulum
Repoyu klonlayın: git clone https://github.com/beratyaalcin/02-memefaceAR.git

Kütüphaneleri kurun: pip install -r requirements.txt

Çalıştırın: python main.py
