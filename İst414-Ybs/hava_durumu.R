#Karar Ağaçları ve ID3 Algoritması (Hava Durumu Tahmini)


# Veri Setini Tanıma ve Değişken Tanımları 

#Bu çalışmada, 1000 gözlemden oluşan clean.csv veri seti kullanılmıştır. Amacımız, hava 
#koşullarına bakarak bir aktivitenin yapılıp yapılmayacağını (Class) tahmin etmektir. 

#Girdi Değişkenleri: 

#• Outlook (Görünüm): Sunny, Overcast, Rain. 
#• Temperature (Sıcaklık): Hot, Mild, Cool. 
#• Humidity (Nem): High, Normal. 
#• Wind (Rüzgar): Weak, Strong. 

#Hedef Değişken (Class): 
# • Yes: Aktivite yapılır. 
# • No: Aktivite yapılmaz. 


##### 1.Veriyi Yükleme ve Ön İşleme ######

# Gerekli kütüphaneler 
library(readr)      #Veri okuma    
library(rpart)      #Karar ağacı modelleme     
library(rpart.plot) #Görselleştirme 
library(caret)      #Performans metrikleri

# 1.Veriyi Oku 
weather_data <- read.csv("C:/Users/esragokce/Desktop/İST414-YBS/hava-durumu.csv")


# 2.Tüm sütunları 'factor' (kategorik) tipine dönüştür 
# ID3, sınıflar arası entropi hesabı yapmak için faktör yapısına ihtiyaç duyar. 
weather_data[] <- lapply(weather_data, as.factor)

# Veri yapısını kontrol et 
str(weather_data)


#### 2.ID3 Modelinin Kurulması #### 
set.seed(123) 

train_index <- createDataPartition( 
  weather_data$Class, 
  p = 0.8, 
  list = FALSE 
) 

train_data <- weather_data[train_index, ] 
test_data  <- weather_data[-train_index, ]

# ID3 Mantığıyla Model Kurma

#rpart fonksiyonunda split = "information" parametresini kullanarak 
#Entropi ve Bilgi Kazancı yöntemini aktif ediyoruz.
weather_model <- rpart( 
  Class ~ .,              # 'Class' hedef, '.' diğer tüm değişkenler 
  data = train_data,  
  method = "class",       # Sınıflandırma problemi 
  control = rpart.control( 
    minsplit = 20,      # Bir düğümde en az 20 örnek varsa dallanma 
    cp = 0.01           # Karmaşıklık parametresi (Budama eşiği) 
  ), 
  parms = list(split = "information") # Bilgi Kazancı (Information Gain) kriteri 
) 

# Model özetini görüntüle 
print(weather_model)



#### 3. Görselleştirme ve Grafik Yorumu #####
# Ağacı okunabilir ve profesyonel bir şekilde çizdiriyoruz. 

# Profesyonel Görselleştirme 
rpart.plot(weather_model,  
           type = 2,           # Karar kuralını düğüm altına yazar 
           extra = 104,        # Olasılık ve gözlem yüzdelerini ekler 
           under = TRUE,       # Metinleri kutu altına yerleştirir 
           cex = 0.8,          # Yazı boyutu (Okunabilirlik için) 
           box.palette = "BuGn", # Yeşil tonları (Yes/No ayrımı için ideal) 
           main = "Hava Durumu Karar Ağacı (ID3)") 



### 4. Model Başarısının Ölçülmesi ###
# Modelin 1000 satır üzerindeki performansını Karışıklık Matrisi ile ölçüyoruz. 

# 1.Model üzerinden tahminleri alalım 
tahminler <- predict( 
  weather_model, 
  test_data, 
  type = "class" 
) 

head(tahminler)

# 2.Confusion Matrix ve Detaylı İstatistikler 

# 'Reference' gerçek değerler, 'Data' tahmin edilen değerlerdir. 
k_matrisi <- table( 
  Gercek = test_data$Class, 
  Tahmin = tahminler 
) 

library(caret) 

confusionMatrix( 
  tahminler, 
  test_data$Class 
)


