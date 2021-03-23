# SETUP

## Linux setup:

Első lépésként a CUDA Toolkit és cuDNN könyvtárak megfelelő verzióját kell telepíteni. Ezekre szükség lesz a TensowFlow telepítéséhez. Fontos, hogy megfelelő verzió legyen telepítve. Az [ITT](https://www.tensorflow.org/install/source#gpu) elérhető táblázatban megtalálható az aktuális kompatibilis verzió.

Installáláshoz a hivatalos guideok: [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)-hoz és [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)-hez

A CUDA verziója ellenőrizhető a `nvcc -V` comanddal, a cuDNN verzióját pedig meg lehet nézni a `cudnn_version.h` fileban (.../cuda/include/cudnn_version.h).

Tesztelés során én a CUDA 11.1 és cuDNN 8.0.4-es verziót használtam.

**Dependenciák installálása**  
A requirements.txt fileban részletesen megtalálható, hogy melyik csomag melyik verzióját használtam.

Új virtuális környezet létrehozása után az alábbi csomagokat kell installálni: 
```
$ pip install tensorflow 
$ pip install dlib
$ pip install face_recognition
$ pip install torch
$ pip install opencv-python
```

Teszteljük le, hogy biztosan elérhető a GPU:
```
$ python3
>>> import tensorflow as tf
>>> tf.test_is_gpu_available()
True
>>> import dlib
>>> dlib.DLIB_USE_CUDA
True
```

## MacOS setup:

**TensorFlow** installálás:
A TensorFlow M1-es Mac-re optimalizált verziója itt érhető el: [LINK](https://github.com/apple/tensorflow_macos). A csomagot Conda environmenten keresztül lehet installálni, amihez van egy részletes leírás [itt](https://github.com/apple/tensorflow_macos/issues/153). Összefoglalva a lépéseket:  
* XCode command line toolok: `xcode-select --install`
* Miniforge telepítése
* új environment létrehozása `environment.yml` file alapján
* TensorFlow installálás:

```
$ pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
```

A további könyvtárak egyelőre nem támogatják az M1-es architektúrát. Condán és Pip-en keresztült lehet felrakni őket:

```
conda install cmake Pillow opencv
conda install pytorch -c isuruf/label/pytorch -c conda-forge
pip install dlib
pip install face_recognition
```


## Google Colab setup:

Fontos, hogy session-nél állítsuk be a GPU használatát. Google colabra a legtöbb csomag már telepítve van alapból, egyedül a face_recognition-t kell külön telepíteni:

```
!pip install face_recognition
```

# TESTS: 

A tesztek futtathatóak az alábbi módon:
```
$ python test_numpy.py
```

* A `test_numpy.py` a Numpy könvyvtár számítási teljesítményét méri. Lefuttatás után kiírja hány másodpercig tartott a script lefuttatása. 
* A `test_mnist.py` a TensorFlow keretrendszert használja, egy classifier betanítását végzi. Megfigyelhető egy tanítási ciklus (epoch) hány másodpercig tart.
* A `test_cifar10.py` Az előzőhöz hasonlóan TensorFlow keretrendszert használ, de itt a modell tanítása hosszabb ideig tart
* A `test_dlib.py` A dlib és face_recognition könyvtárak teljesítményét nézi. 400 képen fut le az arckereső és embedding kinyerő algoritmus.
* A `test_posenet_gpu.py` és `test_posenet_cpu.py` a testtartás elemző PoseNet Pytorch könyvtár furási idejét méri. A GPU verzió kihasználja a grafikus kártyát.