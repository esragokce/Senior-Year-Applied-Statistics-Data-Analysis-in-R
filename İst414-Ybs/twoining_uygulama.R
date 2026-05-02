#Twoing Algoritması - ToothGrowth, mtcars ve Titanic Veri Setleri ile Karşılaştırmalı Analiz

options(encoding = "UTF-8")
Sys.setlocale("LC_ALL", "Turkish")

#Bölüm 1-ToothGrowth Veri Seti


#1.1 Gerekli Paketlerin Yüklenmesi

# rpart: Karar ağacı algoritması (Recursive Partitioning)
# rpart.plot: Karar ağacını görselleştirme
# caret: Sınıflandırma ve regresyon için kapsamlı araç seti
library(rpart)
library(rpart.plot)
library(caret)

#1.2 Veri Setini Yükleme

data(ToothGrowth)
df_tooth <- ToothGrowth
df_tooth$supp <- as.factor(df_tooth$supp)  # Hedef değişkeni faktör yap

# Veri yapısını incele
str(df_tooth)

# Verinin ilk 6 satırını görüntüle
head(df_tooth)

# Hedef değişken: supp (OJ = Portakal suyu, VC = Askorbik asit)
# supp değişkenini faktör olarak düzenle
df_tooth$supp <- as.factor(df_tooth$supp)
# Hedef değişken dağılımı
table(df_tooth$supp)

# len (diş uzunluğu) ve dose (doz) bağımsız değişkenleri
# len: sayısal, dose: sayısal (0.5, 1.0, 2.0)
summary(df_tooth)


#1.3 Model Kurulumu

set.seed(789)
index <- createDataPartition(df_tooth$supp, p = 0.75, list = FALSE)
train_data <- df_tooth[index, ]
test_data <- df_tooth[-index, ]

cat("\n=== VERİ BÖLÜNME SONUÇLARI ===\n")
cat("Eğitim seti:", nrow(train_data), "gözlem\n")
cat("Test seti:", nrow(test_data), "gözlem\n")

cat("\nEğitim seti sınıf dağılımı:\n")
table(train_data$supp)

cat("\nTest seti sınıf dağılımı:\n")
table(test_data$supp)

# Cross validation ayarları (sınıflandırma için)
control_class <- trainControl(
  method = "cv",                      # Cross validation
  number = 5,                         # 5 kat
  classProbs = TRUE,                  # Sınıf olasılıklarını hesapla
  summaryFunction = twoClassSummary,  # ROC, Sens, Spec hesapla
  savePredictions = TRUE
)

# Farklı cp değerleri (grid search)
tune_grid <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1))

# Twoing algoritması ile model eğitimi (5-fold CV)
set.seed(789)
model_tooth_twoing <- train(
  supp ~ .,                      # Hedef: supp, bağımsız: tüm değişkenler
  data = df_tooth,               # Tüm veri seti (CV içinde train/test ayrılacak)    
  method = "rpart",              # Karar ağacı               
  trControl = control_class,     # CV ayarları          
  tuneGrid = tune_grid,          # Farklı cp değerleri               
  parms = list(split = "twoing") # Bölünme kriteri: TWOING        
)

# CV sonuçları tablosu
print(model_tooth_twoing$results)

# ROC grafiği
plot(model_tooth_twoing, main = "ToothGrowth - Twoing Algoritması CV Sonuçları")


final_tree <- model_tooth_twoing$finalModel
rpart.plot(
  final_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = paste("ToothGrowth - Twoing Algoritması\ncp =", model_tooth_twoing$bestTune$cp),
  box.palette = "Greens"
)

#Grafikte tüm cp değerlerinde ROC (AUC) değeri yaklaşık 0.78 civarında sabit görünüyor. Bu, modelin cp değerinden etkilenmediğini gösterir.


# Test setinde tahmin yap
test_pred <- predict(model_tooth_twoing, newdata = test_data)
# Confusion Matrix (Karışıklık Matrisi) oluştur
conf_matrix <- confusionMatrix(test_pred, test_data$supp, positive = "OJ")

print(conf_matrix)



#Bölüm 2- Mtcars Veri Seti

#klasik mtcars veri seti kullanılarak bir aracın özelliklerine (beygir gücü, ağırlık, 
#yakıt tüketimi vb.) bakarak şanzıman tipinin (Otomatik/Manuel) tahmin edilmesi amaçlanmıştır.

library(caret)

data(mtcars)
df_mtcars <- mtcars
df_mtcars$am <- as.factor(df_mtcars$am)           # Hedef değişkeni faktör yap
levels(df_mtcars$am) <- c("Otomatik", "Manuel")


set.seed(789)
index <- createDataPartition(df_mtcars$am, p = 0.75, list = FALSE)
train_data <- df_mtcars[index, ]
test_data <- df_mtcars[-index, ]

cat("\n=== VERİ BÖLÜNME SONUÇLARI ===\n")
cat("Eğitim seti:", nrow(train_data), "gözlem\n")
cat("Test seti:", nrow(test_data), "gözlem\n")

cat("\nEğitim seti sınıf dağılımı:\n")
table(train_data$am)

cat("\nTest seti sınıf dağılımı:\n")
table(test_data$am)


# Twoing algoritması ile model eğitimi (5-fold CV)
set.seed(456)
model_mtcars_twoing <- train(
  am ~ .,                                # Hedef: am, bağımsız: tüm değişkenler
  data = df_mtcars,                      # Tüm veri seti
  method = "rpart",
  trControl = control_class,             # Aynı CV ayarları
  tuneGrid = tune_grid,                  # Farklı cp değerleri
  parms = list(split = "twoing")         # Bölünme kriteri: TWOING
)

print(model_mtcars_twoing$results)


# ROC grafiği
plot(model_mtcars_twoing, main = "mtcars - Twoing Algoritması CV Sonuçları")


# En iyi modelin karar ağacını görselleştir
final_tree_mtcars <- model_mtcars_twoing$finalModel

rpart.plot(final_tree_mtcars, type = 2, extra = 104, 
           main = paste("mtcars - Twoing Algoritması\ncp =", model_mtcars_twoing$bestTune$cp),
           box.palette = "Greens", fallen.leaves = TRUE)


# Test setinde tahmin yap
test_pred <- predict(model_mtcars_twoing, newdata = test_data)

# Confusion Matrix (Karışıklık Matrisi) oluştur
conf_matrix <- confusionMatrix(test_pred, test_data$am, positive = "Manuel")
print(conf_matrix)



##### Bölüm 3-Titanik Veri Seti ################


# titanic: Titanic veri seti için
# rpart: Karar ağacı algoritması (Twoing için)
# rpart.plot: Karar ağacını görselleştirme
# caret: Cross validation ve model değerlendirme için
# dplyr: Veri işleme için
library(titanic)
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)


# 3.1 Veri Hazırlama ve Temizleme

# Titanic veri setini yükle (titanic paketinden)
# titanic::titanic_train - eğitim için hazırlanmış Titanic verisi
mydata <- titanic::titanic_train 

# Verinin yapısını incele
str(mydata)

# Verinin ilk 6 satırını görüntüle
head(mydata)

# Eksik verileri kontrol et
colSums(is.na(mydata))

# Eksik verileri kaldır (NA - boş değer olan satırlar çıkarılır)
# na.omit() fonksiyonu herhangi bir sütunda NA olan satırları siler
mydata <- na.omit(mydata)  



#3.2 Kategorik Değişkenleri Faktör Olarak Tanımla

# Survived: Hayatta kalma durumu (0 = Hayır, 1 = Evet)
# Hedef değişken - tahmin etmek istediğimiz şey
# Twoing için faktör olmalı
mydata$Survived <- as.factor(mydata$Survived)
levels(mydata$Survived) <- c("No", "Yes")  # 0 = No, 1 = Yes

# Pclass: Yolcu sınıfı (1, 2, 3)
# 1. sınıf zenginler, 3. sınıf daha düşük gelirli yolcular
mydata$Pclass <- as.factor(mydata$Pclass)

# Sex: Cinsiyet (male, female) - Titanic'te kadınlar öncelikli kurtarıldı
mydata$Sex <- as.factor(mydata$Sex)

# Embarked: Biniş limanı (C = Cherbourg, Q = Queenstown, S = Southampton)
mydata$Embarked <- as.factor(mydata$Embarked)

# NOT: Age, SibSp, Parch, Fare sayısal değişken olarak kalır
# Twoing algoritması sayısal değişkenleri de işleyebilir


# Değişken tiplerini kontrol et
str(mydata)

# Hedef değişken dağılımı
table(mydata$Survived)

prop.table(table(mydata$Survived)) * 100


# 3.3 Eğitim ve Test Verilerini Ayırma


set.seed(123)  # Sonuçların tekrarlanabilir olması için rastgelelik ayarı

# createDataPartition: Veriyi eğitim/test diye ayırır (sınıf oranlarını korur)
# Survived: Hedef değişken
# p = 0.8: %80 eğitim, %20 test
# list = FALSE: Sonuç indeks vektörü olarak dönsün
trainIndex <- createDataPartition(mydata$Survived, p = 0.8, list = FALSE)

# train_data: Seçilen satırlar (eğitim için)
train_data <- mydata[trainIndex, ]

# test_data: Seçilmeyen satırlar (test için)
test_data <- mydata[-trainIndex, ]

cat("\n=== VERİ BÖLÜNME SONUÇLARI ===\n")
cat("Eğitim seti:", nrow(train_data), "gözlem\n")
cat("Test seti:", nrow(test_data), "gözlem\n")


# Sınıf dağılımlarını kontrol et
cat("\nEğitim seti sınıf dağılımı:\n")
table(train_data$Survived)

cat("\nTest seti sınıf dağılımı:\n")
table(test_data$Survived)


# 3.5 Twoing ile Basit Model (CV’siz)

# =====================================================
# 5.1 TWOING ALGORİTMASI - BASİT MODEL
# =====================================================
# Twoing algoritması ile model eğitimi
# parms = list(split = "twoing"): Bölünme kriteri olarak Twoing kullan
model_twoing <- rpart(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  method = "class",                      # Sınıflandırma problemi
  parms = list(split = "twoing"),        # TWOING bölünme kriteri
  control = rpart.control(
    cp = 0.01,                           # Karmaşıklık parametresi
    minsplit = 10,                       # Minimum bölünme gözlem sayısı
    minbucket = 5                        # Yaprak düğüm minimum gözlem
  )
)
# Model özetini görüntüle
cat("\n=== TWOING MODELİ ÖZETİ ===\n")
print(model_twoing)


# Değişken önem dereceleri
cat("\n=== DEĞİŞKEN ÖNEM DERECELERİ ===\n")
print(model_twoing$variable.importance)


# 3.6 Twoing ile Cross Validation (LOOCV )


# Not: Titanic verisi 700+ gözlem olduğu için 5-fold veya 10-fold kullanılabilir
# Ancak küçük veri setlerinde olduğu gibi LOOCV da uygulanabilir
# LOOCV ayarları
set.seed(123)
control_loocv <- trainControl(
  method = "LOOCV",                      # Leave-One-Out Cross Validation
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Farklı cp değerleri
tune_grid <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1))

# Twoing ile model eğitimi (LOOCV)
model_twoing_loocv <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  method = "rpart",
  trControl = control_loocv,
  tuneGrid = tune_grid,
  parms = list(split = "twoing"),
  metric = "ROC"
)

print(model_twoing_loocv)


# 3.7 Twoing ile 10-Fold Cross Validation

# 10-fold CV ayarları
set.seed(123)
control_10fold <- trainControl(
  method = "cv",                         # Cross validation
  number = 10,                           # 10 kat
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Twoing ile model eğitimi (10-fold CV)
model_twoing_10fold <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  method = "rpart",
  trControl = control_10fold,
  tuneGrid = tune_grid,
  parms = list(split = "twoing"),
  metric = "ROC"
)

print(model_twoing_10fold)


# Her fold'un performansı
print(model_twoing_10fold$resample)

# Modelimi 10 katlı çapraz geçerlilikle zorlu testlerden geçirdim ve performansı bu şekildeydi


# 3.8 KARAR AĞACINI GÖRSELLEŞTİRME

# En iyi modeli seç (10-fold CV ile eğitilen)
best_model <- model_twoing_10fold$finalModel

# Karar ağacını görselleştir
rpart.plot(
  best_model,            
  type = 2,                # Düğüm içi etiket stili          
  extra = 104,             # Sınıf ve yüzde gösterimi           
  fallen.leaves = TRUE,    # Yaprakları aynı hizaya getir 
  main = "Titanic - Twoing Algoritması ile Karar Ağacı\n(Hayatta Kalma Tahmini)",
  box.palette = "GnBu",              # Renk paleti
  cex = 0.8
)


# Alternatif görselleştirme (daha detaylı)
rpart.plot(
  best_model,
  type = 3,
  extra = 106,
  under = TRUE,
  main = "Titanic - Twoing Algoritması (Detaylı)",
  box.palette = "auto"
)


# 3.9 Değerlendirme

# Test setinde tahmin yap
test_pred <- predict(model_twoing_10fold, newdata = test_data)

# Confusion Matrix (Karışıklık Matrisi) oluştur
conf_matrix <- confusionMatrix(test_pred, test_data$Survived, positive = "Yes")

print(conf_matrix)




