# CattleBreedClassifier

Merhaba, ben İsmet. Ailem hayvancılıkla uğraştığı için çocukluğumdan beri bu sektörün içindeyim. Bilgisayar mühendisliği eğitimim boyunca teknoloji ile tarım ve hayvancılığı nasıl birleştirebileceğimi araştırdım. Bu proje de derin öğrenme dersi kapsamında geliştirdiğim, kendi ilgi alanımla akademik çalışmaları bir araya getiren bir sığır ırkı sınıflandırma sistemidir.

## Proje Hakkında

Bu çalışmada 5 farklı sığır ırkını tanıyabilen bir derin öğrenme modeli geliştirdim. Ayrshire, Brown Swiss, Holstein Friesian, Jersey ve Red Dane ırklarına ait görselleri kullanarak çeşitli derin öğrenme mimarilerini test ettim. Amacım farklı modellerin performanslarını karşılaştırmak ve donanım kısıtları altında en verimli modeli bulmaktı.

Geliştirdiğim modeller arasında sıfırdan tasarladığım temel bir CNN modeli ile birlikte VGG16, ResNet50, MobileNetV2 ve EfficientNetB0 yer alıyor. Eğitim süreçlerinde veri artırma tekniklerini ve transfer öğrenme yöntemini kullanarak modellerin başarı oranını artırdım. Yaptığım testler sonucunda en yüksek başarıyı ve en iyi bellek verimliliğini EfficientNetB0 modeli ile elde ettim.

Ayrıca projeye Flask kullanarak modern bir web arayüzü ekledim. Bu arayüz üzerinden sisteme yüklenen yeni sığır görsellerinin hangi ırka ait olduğu anlık olarak analiz edilebiliyor.

## Özellikler

- 5 farklı sığır ırkının sınıflandırılması
- PyTorch ile geliştirilmiş derin öğrenme modelleri
- Model başarımlarının ve eğitim sürelerinin karşılaştırmalı analizi
- Farklı öğrenme hızları ve aşırı öğrenme durumlarının incelenmesi
- Flask tabanlı kullanıcı dostu web arayüzü
- Sürükle bırak destekli görsel yükleme ve tahmin ekranı

## Kurulum ve Kullanım

Projeyi kendi bilgisayarınızda çalıştırmak isterseniz aşağıdaki adımları izleyebilirsiniz.

1. Projeyi bilgisayarınıza indirin:
```bash
git clone https://github.com/kullaniciadi/CattleBreedClassifier.git
cd CattleBreedClassifier
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Web uygulamasını başlatın:
```bash
python app.py
```

4. Tarayıcınızdan `http://localhost:5000` adresine giderek sistemi kullanmaya başlayabilirsiniz.

## Dosya Yapısı

- `app.py`: Web uygulamasının çalıştığı ana Flask sunucu dosyası
- `cnn_assignment.py`: Tüm modellerin eğitildiği, test edildiği ve grafiklerin oluşturulduğu temel Python betiği
- `save_model.py`: En başarılı modeli web arayüzünde kullanmak üzere hazırlayan ve kaydeden betik
- `templates/` ve `static/`: Web arayüzünün tasarımını oluşturan HTML, CSS ve JavaScript dosyaları
- `results/`: Eğitim sonuçlarına ait performans grafikleri, hata matrisleri ve analiz görselleri
- `İsmet_Çelenler_202213709071_DL_S26.pdf`: Projenin detaylı geliştirme adımlarını ve sonuçlarını içeren rapor

## Sonuç

Geliştirdiğim bu proje ile hayvancılık alanındaki saha bilgimi teknik bilgilerle somut bir ürüne dönüştürmeye çalıştım. Sürü yönetim yazılımları ve akıllı tarım teknolojileri alanında kendimi geliştirmeye devam ediyorum. Sistemle veya kullandığım yöntemlerle ilgili fikir alışverişinde bulunmak isterseniz benimle Github üzerinden iletişime geçebilirsiniz.
