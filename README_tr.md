# karşılaşılan zorluk ve hatalar
Genel manada modelin loss kaybını düşürme ve train_accuracy değerini yukarı çekme konusunda problem yaşadım.
yaşadığım problemler:
ilk aşamada modeli 15 epoch ile eğitmeyi denedim

batch normalization ekledik
transforms için resimlere horizontal flip ve random crop ekledik
dropout 0.2
lr 0.01
momentum = 0.9
scheduler step = her 15 epochta bir
2 tane daha conv-bn-relu-pooling katmanı eklendi.
scheduler olarak stepLR yerine learning rate'i daha kavisli düşüren CosineAnnealingLR kullanılmaya başlandı.
