# CUDA ve CuDNN Windowsa nasıl kurulur ?
## CUDA, CuDNN nedir ?
CUDA, "Compute Unified Device Architecture" kısaltmasıdır ve NVIDIA tarafından geliştirilen bir platformdur. CUDA, GPU (Grafik İşlem Birimi) üzerinde genel amaçlı hesaplama işlemleri gerçekleştirmenizi sağlar. Bu, özellikle bilimsel hesaplamalar, veri analizi, derin öğrenme ve benzeri yoğun hesaplama işlemleri için oldukça yararlıdır. CUDA, GPU'yu daha fazla işlem gücü sağlayan paralel hesaplamalar yapmak için kullanmanıza olanak tanır.
CuDNN ise "CUDA Deep Neural Network" kısaltmasıdır ve NVIDIA tarafından geliştirilen bir kütüphanedir. Bu kütüphane, derin öğrenme (deep learning) işlemlerini hızlandırmak için tasarlanmıştır. CuDNN, derin öğrenme çerçeveleri ve uygulamaları için optimize edilmiş GPU üzerinde çalışan hızlı matematiksel işlemler sunar. Bu, yapay sinir ağları ve derin öğrenme algoritmalarının daha hızlı ve verimli bir şekilde eğitilmesine ve çalıştırılmasına olanak tanır.
Özetle, CuDNN, derin öğrenme uygulamalarını hızlandırmak ve GPU üzerinde daha verimli bir şekilde çalıştırmak için kullanılan bir CUDA tabanlı kütüphanedir.

## CUDA, CuDNN’ i bilgisayarımıza nasıl kurarız ?

## 1.Adım: GPU Uyumluluğu Kontrolü
İlk önce bilgisayarımızın sahıp olduğu GPU yacağımız işlemleri destekliyor mu öğrenmemiz gerekiyor. Bunu öğreneceğimiz bağlantı İndirme Bağlantıları adlı klasörde mevcut(https://developer.nvidia.com/cuda-gpus). Bağlantıya gidip kendi GPU’muzu arıyoruz, compute capability öğreniyoruz eğer yoksa Google colab sizleri bekliyor. Aynı işlemleri orada da yapabiliriz. Colab’ ta nasıl derin öğrenme algoritmaları koşturulur başka bir repositories’ de sizlere anlatacağım.

## 2.Adım: GPU Sürücüsü Kurulumu 
Bilgisayarımızın sahip olduğu GPU için driver kuracağız. Driver indirme linkini yine İndirme Bağlantıları dosyasına bıraktım(https://www.nvidia.com/download/index.aspx). Dosyadan linke gidip driver indirip kuruyoruz. Linke gittiğinizde otomatik olarak GPU’ nuzu tanıyarak uygun kısımları dolduracaktır. İndikten sonra next ‘ tıklayarak kurulumu tamamlayın.

## 3.Adım: CUDA Kurulumu
Şimdi CUDA’ yı indirelim. Yine linki yine İndirme Bağlantıları dosyasına bıraktım(https://developer.nvidia.com/cuda-toolkit-archive). Linke tıkladıktan sonra açılan sayfada birçok CUDA versiyonu olacak. Biz bunlardan CUDA Toolkit 11.7.0 (May 2022) sürümünü indireceğiz. Neden son sürümünü kurmadık, çünkü son sürüme uygun CuDNN sürümü mevcut değil. Daha stabil çalışan 11.7.0 sürümü kuracağız. İndirme kısmına gelelim; CUDA Toolkit 11.7.0 (May 2022) kısmına tıklayınca karşımıza çıkan sayfada uygun seçenekleri seçmemiz gerekiyor. İlk olarak işletim sistemi seçiyoruz windows, sonra Architecture x86_64, sonra işletim sistemi versiyonu 10, sonra indirme tipi exe(local) seçiyoruz. Tahmini 2.5, 3 gblık bir exe dosyasını inecektir. İnen dosyaya çift tıklayıp çalıştırıyoruz. Sonra ileri butonuna tıklayarak kurulumu tamamlıyoruz.

## 4.Adım: CuDNN Kurulumu
Şimdi CuDNN’ i indirelim. Yine linki yine İndirme Bağlantıları dosyasına bıraktım(https://developer.nvidia.com/rdp/cudnn-archive). Linke tıkladıktan sonra açılan sayfada birçok CuDNN versiyonu olacak. Biz bunlardan CUDA Toolkit 11.7.0 (May 2022) sürümünü uygun olan CuDNN v8.5.0 (August 8th,2022) for CUDA 11x versiyonun indireceğiz. CuDNN v8.5.0 (August 8th,2022) for CUDA 11x kısmına tıklıyoruz ve Windows için uygun olan Local Installer for Windows(zip) kısmına tıklayarak indirme işlemini başlatıyoruz.

## 5.Adım: Zlib İndirme
Şimdi Zlib’ İ indirelim. Bu dosyayı sizin için yukarı bıraktım. Oradan indirebilirsiniz. Linki İndirme Bağlantıları dosyasında var ama orada bulmanız biraz zor olabilir. O yüzden sizi için yukarı bıraktım.

##6 .Adım: Dosya Düzenleme
Şimdi en önemli kısma geldik. Burada yapacağınız bir hata driverların doğru çalışmasını engeller. Lütfen dikkatli olalım.
İndirdiğimiz CuDNN ve Zlib dosyalarını yerel disk içerisine açtığımız bir dosya içine taşıyalım. CuDNN zip dosyasını buraya ayıklayalım. Yerel disk / Program Files kısmında bulunan NVIDIA GPU Computing Toolkit dosyasını buluyoruz. İçerisine CUDA var. CUDA içerisine giriyoruz. Burada v11.7 adında dosya bulunuyor. Bu klasörün içine giriyoruz. Sonra ayıkladığımız CuDNN klasörünün içine giriyoruz. Bu dosya içinde bin, lib, include adında dosyalar var. ***buradaki dosyaların aynıları CUDA dosyasında da var. şimdi CuDDN / bin dosyasında bulunanları kopyalayıp, CUDA / bin klasörüne yapıştırıyoruz. Yapıştırınca bir uyarı alacaksınız aynı dosyadan burada mevcut uyarısı. Değiştir ve kaydet diyerek devam edin. Aynı şekilde include içindekileri kopyalayıp, yapıştırıyoruz. Aynı şekilde CuDDN / lib dosyasında bulunanları kopyalatıp, CUDA / lib/ x64 içerisine yapıştırıyoruz.
Yerel disk içerisine Zlib adında bir klasör açalım. İçerisine, indirdiğimiz zlib dosyasının içerisindekileri kopyalayıp, yapıştıralım.
Bu oluşturduğumuz Zlib dosyanın yolunu sisteme vermemiz gerekiyor. Windows arama çubuğuna “Sistem ortam değişkenlerini düzenle ” yazıp, açalım. Gelişmiş kısmında Ortam Değişkenleri butonuna tıklıyoruz. Sistem değişkenleri kısmında Path isimli bir değişken var. Bulup çift tıklayarak içine giriyoruz. Sonra acılan pencerede yeni butonuna tıklayın. Sonra gözat butonuna tıklayın. Gözat kısmında, oluşturduğumuz Zlib dosyasını buluyoruz.  Zlib’ e tıklayıp  ve içideki dllx64 klasorünü seçip tamam butonuna tıklıyoruz. Tamam butonlarına tıklayarak değişiklikleri kaydediyoruz. Artık GPU’ muz hazır.

## 7.Adım: Test
Simdi yaptığımız işlemleri test edelim. Ben yolov8 üzerinde denemeyi tercih ettim, siz istediğiniz gibi test edebilirsiniz. Yolo için bir cok kütüphaneye ihtiyacimiz var. Bu yüzden Anaconda ya ihtiyacimiz var ve Anaconda Prompt üzerinde çalışacağız. Anaconda yı kurduğunuzu varsayıyorum. 
Yerel disk’e yolov8 adında bir dosya oluşturalım. Ardında Anaconda Prompt’u açıp cd komutunu kullanarak yolov8 dosya dizinine gelelim. Bu dizinde `$ conda create -n yolov8-gpu python=3.10.6` komutunu çalıştıralım. Ardında `$ conda activate yolov8-gpu` komutunu çalıştıralım. Bu komutlar ile yeni bir environments oluşturup değiştiriyoruz. Terminal yolunuzun yolov8 dosyasında olduğundan emin olduktan sonra gerekli kütüphane kurulumları için sırasıyla `$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --timeout=1000`  , sonra  `$ pip install ultralytics` komutlarını çalıştıralım. Ardında https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt bu linki tıklayarak yada arama motoruna yapıştırarak indirme işlemini yapalım. İnen dosyayı yolov8 dosyasına taşıyalım.
Ardında terminla dizini yolov8 dosyasında  (terminal dizinini cd komutu ile uygun dosyaya getiriniz)  iken `$ yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'` bu komutu çalıştırın.
Herşeyi doğru yaptıysanız yolov8 dosyası içine bir run/detect/predit dosyası oluşacak ve dosya içinde işlenmiş bus.jpg fotoğrafı olması gerekiyor.

## Okuduğunuz için teşekkür ederim umarım işinize yaramıştır….









