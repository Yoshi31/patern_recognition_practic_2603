# Руководство по использованию программы обработки видео с использованием OpenCV
Этот код представляет собой программу для обработки видео с использованием библиотеки OpenCV. Программа читает видеофайл, применяет пороговую обработку в цветовом пространстве HSV, выполняет операции морфологической обработки, находит контуры объектов и рисует прямоугольные ограничивающие рамки вокруг них.

## Инструкции по работе с программой:
Убедитесь, что у вас установлен OpenCV для C++.  

Запустите программу, предварительно указав путь к видеофайлу VID_20240326_103227.mp4.  

После запуска появятся два окна:

"Original Video" для отображения исходного видео с нарисованными прямоугольными ограничивающими рамками.
"HSV Mask" для отображения маски, полученной в результате пороговой обработки в цветовом пространстве HSV.
Вы можете настроить пороговые значения HSV с помощью трекбаров в окне "HSV Mask".

Нажмите q, чтобы завершить выполнение программы.

По завершении работы будут сохранены видеофайлы с нарисованными ограничивающими рамками.

## Примеры работы программы
![image](https://github.com/Yoshi31/patern_recognition_practic_2603/assets/62884580/134adcce-173f-4192-a97b-890f6d6708df)
![image](https://github.com/Yoshi31/patern_recognition_practic_2603/assets/62884580/bb9792d6-cdfa-4fdc-b37a-1671622a258a)
![image](https://github.com/Yoshi31/patern_recognition_practic_2603/assets/62884580/ea818600-98e7-4e59-8f4c-c8e6656ff61c)


