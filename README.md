# Singularis_Test
# Инструкция

Для корректного запуска необходимо установить pytesseract с https://github.com/UB-Mannheim/tesseract/wiki. Устанавливаем в стандартную директорию. После добавляем языковую модель для русского языка (https://tesseract-ocr.github.io/tessdoc/Data-Files#data-files-for-version-400-november-29-2016).

После запуска программы необходимо выбрать: на весь ли экран открыт файл или нет (Это для того, чтобы выбрать нужную кнопку, ведь при окне на весь экран появляется 2 прямоуголика, контуры которых соприкасаются).
Дальше необходимо выбрать одно из пяти тестовых заданий (для третьего теста лучше всего настроить threshold около 229, однако для остальных он тоже будет подходить, но не будет захватывать нижнюю часть с кол-во страниц и т.п.)
Сопоставление иконки и подбор рамки происходит с помощью немного модифицированного mathching template. Если оставить простой метод, то он будет выделять любую рамку В ГРАНИЦАХ КОТОРОЙ находится иконка, но нам нужна именно одна граница, поэтому добавляем сравнения расстояния от иконки и до левого верхнего угла. Также проверяем threshold и threshold предыдущей найденной рамки.
Сопоставление же ключевых элементов происходит обычным методом matching template, только с проверкой на принадлежной верхней строки нужного окна.
