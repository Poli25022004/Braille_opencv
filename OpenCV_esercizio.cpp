#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cctype>

using namespace cv;
using namespace std;

map<string, char> brailleLetters = {
    {"100000",'a'}, {"110000",'b'}, {"100100",'c'}, {"100110",'d'}, {"100010",'e'},
    {"110100",'f'}, {"110110",'g'}, {"110010",'h'}, {"010100",'i'}, {"010110",'j'},
    {"101000",'k'}, {"111000",'l'}, {"101100",'m'}, {"101110",'n'}, {"101010",'o'},
    {"111100",'p'}, {"111110",'q'}, {"111010",'r'}, {"011100",'s'}, {"011110",'t'},
    {"101001",'u'}, {"111001",'v'}, {"010111",'w'}, {"101101",'x'}, {"101111",'y'},
    {"101011",'z'}
};

map<string, int> brailleNumbers = {
    {"100000",1}, {"110000",2}, {"100100",3}, {"100110",4}, {"100010",5},
    {"110100",6}, {"110110",7}, {"110010",8}, {"010100",9}, {"010110",0}
};

const string BRAILLE_CAPITAL = "000001";
const string BRAILLE_NUMBER = "001111";

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Errore apertura webcam\n";
        return -1;
    }

    Mat frame, gray, bin;
    string lastSaved;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
                                                                                                                                                                                             
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);
        adaptiveThreshold(gray, bin, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(bin, bin, MORPH_OPEN, kernel);
        morphologyEx(bin, bin, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Point2f> dots;
        for (auto& c : contours) {
            double area = contourArea(c);
            if (area < 30 || area > 400) continue;

            Rect box = boundingRect(c);
            float aspect = (float)box.width / box.height;
            if (aspect < 0.5f || aspect > 1.5f) continue;

            Point2f center;
            float radius;
            minEnclosingCircle(c, center, radius);

            dots.push_back(center);
            circle(frame, center, 6, Scalar(0, 0, 255), 2);
        }

        if (dots.empty()) {
            imshow("Braille", frame);
            imshow("Binary", bin);
            if (waitKey(30) == 27) break;
            continue;
        }

        sort(dots.begin(), dots.end(),
            [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

        vector<vector<Point2f>> rows;
        vector<Point2f> row;
        const float rowThresh = 25.f;

        for (auto& p : dots) {
            if (row.empty() || abs(p.y - row.back().y) < rowThresh)
                row.push_back(p);
            else {
                rows.push_back(row);
                row.clear();
                row.push_back(p);
            }
        }
        if (!row.empty()) rows.push_back(row);
          
        string finalText;

        bool numberMode = false;
        bool capitalizeNext = false;

        for (auto& r : rows) {
            sort(r.begin(), r.end(),
                [](const Point2f& a, const Point2f& b) { return a.x < b.x; });

            vector<vector<Point2f>> cells;
            vector<Point2f> cell;
            const float cellGap = 40.f;

            for (size_t i = 0; i < r.size(); i++) {
                if (i == 0 || abs(r[i].x - r[i - 1].x) < cellGap)
                    cell.push_back(r[i]);
                else {
                    cells.push_back(cell);
                    cell.clear();
                    cell.push_back(r[i]);
                }
            }
            if (!cell.empty()) cells.push_back(cell);

            for (size_t i = 0; i < cells.size(); i++) {
                auto& c = cells[i];

                float minX = 1e9f, maxX = 0.f;
                float minY = 1e9f, maxY = 0.f;

                for (auto& p : c) {
                    minX = min(minX, p.x);
                    maxX = max(maxX, p.x);
                    minY = min(minY, p.y);
                    maxY = max(maxY, p.y);
                }

                float h = maxY - minY;
                if (h < 5) continue;
                char code[7] = "000000";
                float cellW = maxX - minX;
                float cellH = maxY - minY;

                for (auto& p : c) {
                    int col = (p.x - minX) < cellW / 2 ? 0 : 1;
                    int row = (p.y - minY) < cellH / 3 ? 0 :
                        (p.y - minY) < 2 * cellH / 3 ? 1 : 2;
                    int idx = col * 3 + row; // 0..5
                    code[idx] = '1';
                }

                string key(code);

                if (i > 0) {
                    float gap = c[0].x - cells[i - 1][0].x;
                    float cellWidth = maxX - minX;
                    if (gap > 2.2f * cellWidth) {
                        finalText += ' ';
                        numberMode = false;
                        capitalizeNext = false;
                    }
                }
                if (key == BRAILLE_CAPITAL) {
                    capitalizeNext = true;
                    continue;
                }

                if (key == BRAILLE_NUMBER) {
                    numberMode = true;
                    continue;
                }

                char out = 0;

                if (numberMode && brailleNumbers.count(key)) {
                    out = '0' + brailleNumbers[key];
                }
         
                else if (brailleLetters.count(key)) {
                    out = brailleLetters[key];
                    numberMode = false;
                }
                else {
                    continue;
                }

                if (capitalizeNext) {
                    out = toupper(out);
                    capitalizeNext = false;
                }

                finalText += out;
            }

            finalText += ' ';
            numberMode = false;
            capitalizeNext = false;
        }

        putText(frame, finalText, Point(20, 40),
            FONT_HERSHEY_SIMPLEX, 1,
            Scalar(255, 255, 255), 2);

        if (!finalText.empty() && finalText != lastSaved) {
            ofstream f("frase_braille.txt", ios::app);
            f << finalText << endl;
            lastSaved = finalText;
        }

        imshow("Braille", frame);
        imshow("Binary", bin);

        if (waitKey(30) == 27) break;
    }
    return 0;
}