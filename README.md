# Тестовое задание

### Инструкция по обучению нейросети:

* Положить в папку с ноутбуком папку *internship_data* с папками *female* и *male*. В папках *female* и *male* должны быть фото соответственно женщин и мужчин.


* Запустить все ячейки Jupyter-ноутбука *Test_task*.

### Инструкция по запуску нейросети:

* В папке с моделью *internship_model* и файлом *test_model.py* запустить командную строку. В командной строке ввести "python test_model.py '*путь к папке с фото*'", нажать Enter. Скрипт сохранит результат в файле *process_results.json*.

### Описание решения:

* Данные были подготовлены составителем задания, затем были созданы папки с обучающей, валидационной и тестовой выборкой. Затем данные были трансформированы в датасеты pytorch. Для обучающей выборки трансформации включали в себя не только изменение размера, преобразование в тензор и нормализацию, но и аугментацию - горизонтальное отзеркаливание. Для остальных выборок все кроме аугментации.


* Преобразование размера были необходимо так как это необходимо для корректной работы предобученной сети. Нормализация тензоров осуществлялась так, чтобы на вход сети подавались примеры подобные тем на которой она была обучена.


* Предобученная модель довольно сложна, чтобы ее целиком здесь описывать. Сама по себе это сверточная нейронная сеть. В ней применяются *Batch-Normalization*, пулинг, функция активации PReLU, слой Dropout ближе к выходу сети. В сети огромное количество слоев. (IResNet34 - https://github.com/deepinsight/insightface, в задании использован порт на pytorch: https://github.com/nizhib/pytorch-insightface.)


* Использовался довольно стандартный, но тем не менее хороший алгоритм оптимизации *Adam*, скорость обучения бралась стандартной 0.001, также был создан планировщик, уменьшающий скорость обучения каждые 7 эпох.


* В результате на тестовой выборке был получен *accuracy* 0.97755.