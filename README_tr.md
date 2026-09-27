>Dil: Türkçe 🇹🇷<br>

İngilizce için: [İngilizce](README.md)
![CNN-](https://shieldcn.dev/header/surface.svg?title=CNN-Siniflandirma+modeli&subtitle=cifar-10+ile+egitildi&logo=lu%3ALayers&mode=dark&theme=orange&font=jetbrains-mono&image=https%3A%2F%2Fimages.unsplash.com%2Fphoto-1550745165-9bc0b252726f%3Fw%3D1600%26q%3D70%26fit%3Dcrop%26fm%3Djpg&overlay=0.80)

# Tanım
 CNN (Evrişimli Sinir Ağları) ile ilgili olan bir başka Derin Öğrenme projesini sunmaktan gurur duyuyorum. Bu projede eğitilen model, an itibariyle görsellerdeki 10 farklı sınıftan nesneleri tahmin edebiliyor.<br>

 Bu kez yalnızca klasik ANN kullanmıyoruz. Çünkü model karmaşık görsellerin özelliklerini öğrenmek için daha karmaşık bir yapıya ihtiyaç duyuyor. CNN, işte tam bu yüzden kullanılıyor. Modelin çalışma mantığını anlamak için gerekli her bir detayı bu projede bulacaksınız. Ayrıca, CNN yapısı proje dahilinde net bir şekilde açıklanmıştır.<br>

---

# CNN (Evrişimli Sinir Ağları) nedir?
 CNN'in tanımı için bir benzetme kullanmak zorunda kalsaydım, CNN'in, ANN'in abisi olduğunu söylerdim. Çünkü CNN, basitçe ANN'in daha karmaşık bir versiyonu. Son aşamada klasik ANN içermesine rağmen, CNN'i farklı kılan şey, eklenen evrişim katmanlarıdır. Bu katmanlar görsellerden özellik çıkarmak için kullanılır. Bu sayede, ANN rastgele pikseller yerine, öğrenilmiş özellikleri girdi olarak alır.<br>

![Cnn](assets/cnn-visualization.png "CNN Mantığı")
>Görsel 1.1: CNN Mantığı

## CNN Öğrenme Adımları
Öğrenme adımları iki ana aşamadan oluşur:
- **Özellik Çıkarma Aşaması** (Evrişim Katmanları)
- **Sınıflandırma Aşaması** (Klasik ANN)

### 1- Özellik Çıkarma Aşaması
Özellik çıkarma aşaması dört adımdan oluşur:
- **Evrişim Filtreleri**: Özellik çıkarma aşaması kernel olarak adlandırılan evrişim filtrelerini içerdiği için, en kritik olandır. Her bir kernel kendi işlemlerinden sonra bir özellik haritası oluşturur. Genelde 3x3'lük matris halindedirler. Her bir matrisin görevi gelen görselde (32x32) belirli bir özellik aramaktır (dikey çizgiler gibi). Evrişim filtresi ağırlıklarını her bir resmin üç kanalındaki (RGB) 3x3'lük alana uygular. Daha sonra, üç ayrı sonucu tek bir sayısal değer elde etmek için toplar. Bu değer, o belirli alanda aranan özelliğin (dikey çizgiler) ne kadar baskın olduğunu işaret eder. Her bir filtre, kendi özellik haritasındaki değerleri doldurana kadar, 32x32 boyutunda olan görsellerdeki her bir 3x3 boyutunda alana aynı şeyi yapar.<br>

- **Batch Normalizasyonu**: Aynı filtre batch içindeki her bir görsele uygulanır. 64 görsel içeren bir batch için, tek bir spesifik filtrenin oluşturduğu 64 özellik haritasının tamamını toplar. Batch normalizasyonu sadece 64 görseli kullanmaz, aynı zamanda bu özellik haritalarındaki tüm pikselleri (yükseklik ve genişlik) de dahil eder. O spesifik özellik (dikey çizgiler gibi) için tek bir ortalama ve varyans hesaplamak amacıyla bütün bu değerleri birbiriyle toplar. Daha sonra, bu ortalama ve varyansı kullanarak her bir sayısal değeri normalize eder. Son olarak, ağın doğrusal olmayan özelliklerini korumak için iki tane öğrenilebilir parametre (scale ve shift) uygular. 32 özellik haritasının her biri, evrişim aşamasından sonra bu aynı işlemi birbirinden bağımsız olarak yapar.
- **Aktivasyon Fonksiyonu (ReLU)**: 32 özellik haritasındaki her bir değeri alır ve negatif olanları sıfır (0) ile değiştirir. Pozitif olan değerleri değiştirmez. Basitçe, ReLU'dan sonra özellik haritalarının içindeki en düşük değer sıfır (0) olur.
- **Pooling**: Özellik haritalarının piksellerinin yarısını keser (Özellik haritalarını yeniden boyutlar). Pooling'in iki varyasyonu vardır. İkisi de 2x2 boyutunda piksel adımlarıyla (stride=2) 2x2 boyutunda kerneller kullanır:
    - **Max Pooling**: 2x2'lik alandaki en yüksek değeri alır.
    - **Average Pooling**: 2x2'lik alandaki dört değerin ortalamasını alır.

Bu dört adım özellik çıkarma aşamasındaki her bir katman için art arda uygulanır. 2-4 döngü genel olarak yeterlidir. İhtiyaç halinde daha fazlası uygulanabilir. Her döngüden sonra özellik haritalarının sayısı ikiye katlanırken, boyutları yarı yarıya küçülür.
![Conv](assets/conv-layer.png "Conv-BN-ReLU-Pooling adımları")
>Görsel 1.2: Conv-BN-ReLU-Pooling adımları.

### 2- Sınıflandırma Aşaması (Tam Bağlantılı Katmanlar)
Sınıflandırma aşaması, ANN ile aynı yapıyı kullanır. ANN girdi olarak tek boyutlu vektörler aldığı için, son evrişim döngüsünden sonra flatten (tek boyuta indirgeme) uygulamak zorundayız:
  - '*Flatten*' nedir ve neden uyguluyoruz ?
    - Her bir evrişim döngüsü, dört boyutlu bir tensor döndürür. Örneğin, üçüncü döngüden sonra şu tensoru elde ederiz:  [64,128,8,8]<br> 
    Bu tensor'de;<br> 
    İlk indeks➡️ batch boyutunu işaret eder<br>
    İkinci indeks ➡️ anlık özellik haritalarının sayısını işaret eder<br>
    Son iki indeks ➡️ özellik haritalarının o anki şeklini işaret eder. (8x8)<br>

Dört boyut içeren bu tensor ile, ANN (karar verici) çalışamaz. Çünkü ANN input olarak sadece tek boyutlu vektörler alır. Flatten fonksiyonunun syntax ve parametreleri [classes.py](classes.py) dosyasından gözlemlenebilir.<br>
![Classification](assets/classification-phase.png "Classification phase") 
>GÖrsel 1.3: Sınıflandırma aşaması

- Ağın geri kalanı karışık bir şey içermez. Klasik ANN yapısı takip eder;
  - 1D vektör➡️ Tam bağlantılı katman ➡️ ReLU ➡️ Dropout <br>
>[!NOTe]
> Genel olarak, modelin özellik çıkarma aşaması yeterince becerikli oluşturulduğu sürece, iki tane tam bağlantılı katman modelin karar vermedeki doğruluğunu beslemek için yeterlidir . Dropout tam bağlantılı katmanlarda opsiyoneldir. Ama *overfitting* önlemi olarak dropout kullanmak genellikle bir sektör standardıdır.

## Model Eğitim Detayları
Eğitim aşaması sırasında hiçbir kullanıma hazır model kullanılmamıştır. Her adım özgün bir şekilde atılıp, verilen `.py` dosyalarında görülebileceği üzere, model sıfırdan inşaa edilmiştir.<br>

**Model**:`Evrişimli Sinir Ağları`<br>
**Görev**:`Nesne Sınıflarını Tanıma`<br>
**Epoch**:`30`<br>
**Veri Seti**: 🔍[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html)<br>
**Girdi Görseli Boyutu**`:32x32`<br>
**Teknoloji Yığını**:![NumPy](https://img.shields.io/badge/NumPy-black?style=flat-square&logo=numpy&logoColor=%23db7d25
),![OpenCV](https://img.shields.io/badge/OpenCV-black?style=flat-square&logo=opencv&logoColor=%239cdb25
),![Python](https://img.shields.io/badge/Python-black?style=flat-square&logo=python&logoColor=%231ddb4f
),![PyTorch](https://img.shields.io/badge/PyTorch-black?style=flat-square&logo=pytorch
)<br>
>[!IMPORTANT]
> CIFAR-10 veri seti, direkt kullanılmamış olup bir dizi ön işleme, veri yükleme aşaması sırasında veri setine uygulanmıştır:
```python
 transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(((0.5,0.5,0.5)),(0.5,0.5,0.5))
    ])
```
- Bu dönüştürme işlemleri, veri setine augmentation uygulayarak *overfitting*'i önlemek için kritik öneme sahiptir.

# Model Çıktıları
Model tarafından daha önce görülmemiş bir görsel ile modelin doğruluğunu değerlendirmek için, modele bir uçak görseli verilmiştir.
<div align="center">
  <h4>TEST GÖRSELİ</h4>
  <img src="assets/akinci.png" alt="Description">
</div>
<div align="center">
  <h4>MODEL ÇIKTISI</h4>
  <img src="assets/model-output.png" alt="Description">
</div>

## Performans Değerlerndirmesi
- Model 30 epoch boyunca eğitilmiştir. İlk denemelerde *overfitting* yaşadığım için, uygun epoch sayısını bulmak biraz zamanımı aldı. Aslında *overfitting*'den kurtulmak için farklı parametreleri değiştirmeyi denedim. Takip eden başlıkta, karşılaştığım bütün engelleri göreceksiniz. Mücadele, *underfitting* ile *overfitting* arasındaki mükemmel dengeyi bulmaktı. En düşük loss değerini sağlayan en iyi parametreleri bulduğumda, CNN modellerinin doğruluk değerleri anlamında bir sınırı olduğunu fark ettim. İşte şu ana kadar ulaştığım en iyi değerler:<br>

![Modelin tepe noktası](assets/accuracy-values.png)
![Zaman içinde training loss](assets/training-loss-graph.png)

- Main.py dosyasının içerisinde akıl etmesi çok zor olmayan, bir çeşit *overfitting* dedektörü kullandım:
```python
# 8 - Muhtemel  overfitting tespiti için iki değeri de kontrol et.
    print(f"Train Accuracy: {train_accuracy} % \n Test Accuracy: {test_accuracy} % ")
    if train_accuracy-test_accuracy>10:
        print(f"Possible Overfitting has been detected. You might consider changing epoch number.")
```
ve print() fonksiyonunun içerisindeki mesaj erken eğitim aşamasında tahmin ettiğimden daha fazla belirdi. Problemleri ve denediğim çözümleri, bir sonraki başlığın altında analiz edebilirsiniz.

#### Engeller ve hatalar
Öncelikle, bu projede AI modelleri ile ilgili bir çok yeni detay öğrendim. *Overfitting* ve *underfitting*'e karşı bir çok farklı çözüm kullandım.
- *Underfitting*'e sebep olan 15 epoch ile eğitime başladım. Maksimum doğruluk değerleri %60-70 arasında idi ve bu epey yetersizdi. Daha sonra 25 epoch yapmayı denedim ve bu *overfitting*'e sebep oldu. Model eğitim veri setini gayet iyi öğrenmişti ama eğitim ve test doğruluğu arasındaki makas, %10-15 ile pik yapmıştı.
- Daha sonra, o zamana kadar kullanmadığım batch normalizasyonunu eklemeye karar verdim. Bu modelin öğrenmesini hızlandırdı ama *data augmentation* uygulamaya karar veresiye kadar çözemediğim daha ciddi bir *overfitting* problemine neden oldu.
- BN, evrişim katmanlarına uygulandıktan sonra *overfitting* en büyük problem olduğu için, eğitim veri setini çeşitlendirmek amacıyla burada: [Model Training Details](#model-training-details) daha önce bahsedilen bazı *data augmentation* fonksiyonlarını kullanmak zorunda kaldım.
- Bununla birlikte, model *overfitting*'in artık büyük bir problem olmadığı daha güçlü bir pozisyona yerleşmişti. Bir sonraki hedef, *Data augmentation*'dan sonra modelin daha büyük bir kapasiteye sahip olduğunu düşündüğüm için, mümkün olan en düşük loss değerine ulaşmaktı. Bunu başarmak için, modelin kapasasitesini sınırlandırdığını düşündüğüm iki parametrenin değerlerini değiştirdim:  
  - Dropout = 0.5 > 0.2
  - lr = 0.01 > 0.1

- Bu iki güncelleme, daha önce deneyimlememiş olduğum bir probleme sebep oldu. Bu yüksek öğrenme oranlarından kaynaklanan *overshotting* idi. Bu durumda, model ağırlıklarını devasa bir şekilde günceller ve mükemmel ağırlık değerlerini kaçırır. Loss değeri 2.30'da kalmış ve düşmüyordu. Bununla başa çıkmak için, lr değerini tekrardan 0.01 ile değiştirdim. Dropout 0.2 değeri hata değerini düşürmede iyi iş çıkarıyordu ama hala ufak bir *overfitting* vardı.
- Bu yüzden *lr* değerini bir noktadan sonra yarı yarıya düşürmek için *StepLR* algoritması kullanan bir *scheduler* kullanmaya başladım. Bu son kalan *overfitting* problemini çözdü. Ama daha ileri gidebileceğimi biliyordum.  

- Modeli bir sonraki seviyeye çıkarmak için, iki tane daha evrişim katmanı (conv-bn-relu-pooling) ekledim. Bu çok işime yaradı ve model görselden daha fazla özellik çıkarmaya başladı. Tahmin edebileceğiniz üzere, training loss değeri büyük ölçüde düşerken, doğruluk değerleri epey bir arttı.
- Son dokunuş için, scheduler olarak *CosineAnnealingLR* kullanmaya başladım. Bu bana öğrenme oranına ani bir düşüş uygulamak yerine, öğrenme oranını uçağın iniş yapması gibi yumuşak bir şekilde yavaşça düşürerek yardım etti. Bu sayede, hata değeri aşamalı bir şekilde daha güvenli bir seviyeye yerleşti.<br>

Son olarak, en güvenilebilir loss ve accuracy değerlerine ulaştım [Performance Evaluation](#performance-evaluation). Bu seviyeler ayrıca, önceden eğitilmiş bir model kullanmadan, CNN tabanlı model için ulaşılabilecek limit olarak bilinir.

#### CNN için bir sonraki durak
Merakımı dindirmek için, Transfer Learning kullanarak, önceden eğitilmiş bir modeli CNN'in en gerçek limitini görmek için projeye ekleyeceğim.

## Kurulum ve kullanım
Bu repoyu yerel bilgisayarınıza kopyalayarak bir şans verebilir ve modelin 10 farklı sınıfı tahmin etmede ne kadar tutarlı olabileceğini görebilirsiniz.<br>
1-
```bash
#Repoyu yerel bilgisayarınıza kopyalayın.
git clone https://github.com/halileroglu711/CNN-classification.git
```
2-
```bash
#Proje dosyasına girin.
cd CNN-classification
```
3-
```bash
#Gerekli Kütüphaneleri yükleyin.
pip install -r requirements.txt
```
4-
```bash
#Loss ve accuracy değerlerini kendiniz görmek için eğitimi başlatın.
python main.py
```
5-
```python
#Modeli test edin.
python test.py
```
>```Modeli kendi görselinizle test edebilirsiniz. Sadece Image.open() parametresini kendi görselinizle değiştirin: img=Image.open("assets/akinci.png").convert("RGB") ```

### Proje Dosya Yapısı
```text
CNN-classification/
│
├── assets/
│     ├──cnn-visualization.png
│     ├──conv-layer.png
│     ├──classification-phase.png
│     ├──akinci.png
│     ├──model-output.png
│     ├──accuracy-values.png
│     ├──random-sample-images.png
│     └──training-loss-graph.png
│
│
├── weights/
│      └── best.pt
│
├── .gitignore
├── classes.py
├── functions.py
├── main.py
├── README_tr.md
├── README.md
├── requirements.txt
└── test.py
```
### 💼 License
Bu proje **MIT Lisansı** ile lisanslıdır. <br>
Daha fazla bilgi için [LİSANS](LICENSE) dosyasına göz atın.


### 📬 İletişim
- Herhangi bir hatam varsa bana bildirin🙋. Katkılarınızı bekliyorum 🙂.Bana buradan ulaşabilirsiniz:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/halil-ero%C4%9Flu-5505783a1)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/halileroglu711)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:halileroglu711@gmail.com)






