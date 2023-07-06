import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import relevanssi

# Veri seti örnekleri
veri = [
    ('Bu film harika bir deneyimdi.', 'olumlu'),
    ('Bu kitap sıkıcıydı.', 'olumsuz'),
    ('Bu restoranın yemekleri lezzetliydi.', 'olumlu'),
    ('Bu otelin hizmeti çok kötüydü.', 'olumsuz')
]

# Verileri özniteliklere ve hedef etiketlere ayırma
dokumanlar, etiketler = zip(*veri)

# Tokenizasyon ve kökleri çıkarma
lemmatizer = WordNetLemmatizer()
dokumanlar = [' '.join([lemmatizer.lemmatize(kelime) for kelime in word_tokenize(dokuman.lower())]) for dokuman in dokumanlar]

# Özellik vektörlerini oluşturma
vectorizer = TfidfVectorizer()
ozellikler = vectorizer.fit_transform(dokumanlar)

# Sınıflandırma modelini eğitme
model = SVC()
model.fit(ozellikler, etiketler)

# Relevanssi entegrasyonunu yapılandırma
relevanssi_baslat()
relevanssi_ayarlar = relevanssi_get_settings()
relevanssi_ayarlar['ranking'] = ['custom'] # 'custom' değerini listenin içine ekleyin
relevanssi_ayarlar['ranking_custom'] = ['yapay_zeka_skoru'] # 'yapay_zeka_skoru' değerini listenin içine ekleyin
relevanssi_set_settings(relevanssi_ayarlar)

# Arama sorgusu için yapay zeka modelinden tahmin alma ve sonuçları Relevanssi'ye aktarma
arama_sorgusu = 'harika bir film'
tahmin = model.predict(vectorizer.transform([arama_sorgusu]))[0]

# Relevanssi'ye yapay zeka skorunu aktarma
relevanssi_ayarlar = relevanssi_get_settings()
relevanssi_ayarlar['metadata'] = [{'post_id': 1, 'yapay_zeka_skoru': 0.9}]
relevanssi_set_settings(relevanssi_ayarlar)

# Relevanssi'ye arama sorgusu ve skorunu gönderme
relevanssi_search(arama_sorgusu)

# Relevanssi tarafından döndürülen sonuçları alma
sonuclar = relevanssi_get_results()

# Sonuçları yazdırma
for sonuc in sonuclar:
    print(sonuc['title'], sonuc['score'])

