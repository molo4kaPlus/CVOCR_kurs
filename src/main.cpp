#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>

using namespace std;
using namespace cv;

// Функция для выделения красного цвета
Mat detectRedColor(const Mat& frame) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Диапазон красного цвета в HSV
    Mat mask1, mask2;
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1); // Красный диапазон 1
    inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2); // Красный диапазон 2

    // Объединяем маски
    Mat redMask;
    bitwise_or(mask1, mask2, redMask);

    // Применяем маску к исходному изображению
    Mat redOnly;
    bitwise_and(frame, frame, redOnly, redMask);

    return redOnly;
}

int main() {
    VideoCapture cap("camera.mp4");

    if (!cap.isOpened()) {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Инициализация Tesseract
    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "eng")) { // Используйте "rus" для русского языка
        cout << "Could not initialize tesseract" << endl;
        return -1;
    }

    // Устанавливаем режим распознавания только цифр
    tess.SetVariable("tessedit_char_whitelist", "0123456789");

    while (1) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) { break; }

        // Обрезаем кадр (если нужно)
        frame = frame(Range(500, 800), Range(1000, 1500));

        // Выделяем красный цвет
        Mat redOnly = detectRedColor(frame);

        // Преобразуем в оттенки серого
        Mat gray;
        cvtColor(redOnly, gray, COLOR_BGR2GRAY);

        // Применяем бинаризацию
        Mat binary;
        threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // Распознаем текст с помощью Tesseract
        tess.SetImage(binary.data, binary.cols, binary.rows, 1, binary.step);
        char* outText = tess.GetUTF8Text();
        cout << "Recognized text: " << outText << endl;

        // Получаем bounding boxes для распознанных цифр
        tesseract::ResultIterator* ri = tess.GetIterator();
        if (ri != 0) {
            do {
                const char* word = ri->GetUTF8Text(tesseract::RIL_WORD);
                if (word != nullptr) {
                    int x1, y1, x2, y2;
                    ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2);

                    // Рисуем зеленую рамку вокруг цифры
                    rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                    delete[] word;
                }
            } while (ri->Next(tesseract::RIL_WORD));
            delete ri;
        }

        delete[] outText;

        // Отображаем результат
        imshow("Red Digits", frame);

        char c = (char)waitKey(25);
        if (c == 27) { break; } // Выход по нажатию ESC
    }

    cap.release();
    destroyAllWindows();

    return 0;
}