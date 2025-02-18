# Face recognition system for smart office. 

Идея заключается в том, что бы распознавать лицо, кидать его на сервер и там уже делать полное распознавание по сотрудникам. 

## 1

Начать стоит из файла `create_dataset.py`. Он нужен для создания датасетов, фото создаются по нажатию кнопки `k`. 
Для хорошей работы необходимо иметь больше 50 изображений на каждого.

На заметку: вряд ли OpenCV будет хорошо работать на RasberryPI, так как оно жрет CPU. 

## 2

Как обучить и запустить face detection? 

[полезная ссылка 1](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/), [полезная ссылка 2](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

1) Создать папку в `dataset/` с именем пользователя. Имя папки это и есть имя пользователя, которое будет отображаться на экране.

2) Запустить файл `create_dataset.py`, который поможет создать dataset. Обязательно нужно создать и указать папку пользователя в которую запишется dataset.

3) Запустить файл `extract_embeddings.py`. В кратце: он обработает и запишет датасет всех пользователей в директорию `opencv-recognition/output`.  

4) Запустить файл `train_model.py`. 

5) После обучения, запустить `app.py` и `server.py`.

Все работает отлично, для распознавания на 90% - 100% желательно иметь датасет из 50+ картинок. Но как минус стоит отметить, что оно жрет много ресурсов.

## Планы

Нужно бы испытать и попробовать оптимизировать данную систему для RasberryPI, что бы сама расбери не сгорела от нагрузки. 
Так же в планах изучить подробней использование нейросетей в OpenCV.
