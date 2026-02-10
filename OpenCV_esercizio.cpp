#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cfloat>
#include <cmath>
using namespace cv;
using namespace std;

map<int, char> brailleLetters = {
    {1,'A'}, {5,'B'}, {3,'C'}, {11,'D'}, {9,'E'}, {7,'F'}, {15,'G'}, {13,'H'},
    {6,'I'}, {14,'J'}, {17,'K'}, {21,'L'}, {19,'M'}, {27,'N'}, {25,'O'}, {23,'P'},
    {31,'Q'}, {29,'R'}, {22,'S'}, {30,'T'}, {49,'U'}, {53,'V'}, {46,'W'}, {51,'X'},
    {59,'Y'}, {57,'Z'}
};

map<int, int> brailleNumbers = {
    {1,1}, {5,2}, {3,3}, {11,4}, {9,5}, {7,6}, {15,7}, {13,8}, {6,9}, {14,0}
};

map<int, char> brailleSymbols = {
    {4, ','}, {46, '.'}
};

const int BRAILLE_CAPITAL = (1 << 5);
const int BRAILLE_NUMBER = (1 << 4) | (1 << 1) | (1 << 3) | (1 << 5);

int buildMask(int mat[3][2]) {
    int mask = 0;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            if (mat[r][c]) mask |= (1 << (r * 2 + c));
    return mask;
}

int decodeCellMask(const vector<Point2f>& points) {
    int mat[3][2] = { 0 };
    if (points.empty()) return -1;
    float minX = points[0].x, maxX = points[0].x;
    float minY = points[0].y, maxY = points[0].y;
    for (const auto& p : points) {
        minX = min(minX, p.x);
        maxX = max(maxX, p.x);
        minY = min(minY, p.y);
        maxY = max(maxY, p.y);
    }
    float padX = max(2.0f, (maxX - minX) * 0.1f);
    float padY = max(2.0f, (maxY - minY) * 0.1f);
    minX -= padX; maxX += padX;
    minY -= padY; maxY += padY;
    float colMid = minX + (maxX - minX) / 2.0f;
    float row1 = minY + (maxY - minY) / 3.0f;
    float row2 = minY + 2.0f * (maxY - minY) / 3.0f;
    for (const auto& p : points) {
        int col = (p.x < colMid) ? 0 : 1;
        int row = (p.y < row1) ? 0 : (p.y < row2 ? 1 : 2);
        mat[row][col] = 1;
    }
    return buildMask(mat);
}

Rect rectFromPoints(const vector<Point2f>& pts) {
    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
    for (const auto& p : pts) {
        minX = min(minX, p.x);
        minY = min(minY, p.y);
        maxX = max(maxX, p.x);
        maxY = max(maxY, p.y);
    }
    return Rect(Point((int)minX, (int)minY), Point((int)maxX + 1, (int)maxY + 1));
}

int levenshtein(const string& s1, const string& s2) {
    int n = (int)s1.size(), m = (int)s2.size();
    if (n == 0) return m;
    if (m == 0) return n;
    vector<int> prev(m + 1), cur(m + 1);
    for (int j = 0; j <= m; ++j) prev[j] = j;
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        for (int j = 1; j <= m; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            cur[j] = min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
        }
        prev.swap(cur);
    }
    return prev[m];
}

string normalizeText(const string& s) {
    string out;
    for (char ch : s) {
        if (ch == '\r' || ch == '\n') continue;
        out.push_back(toupper((unsigned char)ch));
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    Mat frame, gray, bin;
    string lastText = "";
    ofstream out("frase_decodificata.txt");
    const string targetPhrase = "PANTORC #40 MG COMPRESSE";
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);
        adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(bin, bin, MORPH_OPEN, kernel);
        morphologyEx(bin, bin, MORPH_CLOSE, kernel);
        vector<vector<Point>> contours;
        findContours(bin.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        double maxArea = 0;
        int maxIdx = -1;
        for (size_t i = 0; i < contours.size(); ++i) {
            double a = contourArea(contours[i]);
            if (a > maxArea) { maxArea = a; maxIdx = (int)i; }
        }
        Mat warped = frame.clone();
        Mat warpedBin = bin.clone();
        Size warpSize(800, 1100);
        if (maxIdx >= 0 && maxArea > (frame.cols * frame.rows) * 0.05) {
            vector<Point> approx;
            approxPolyDP(contours[maxIdx], approx, arcLength(contours[maxIdx], true) * 0.02, true);
            if (approx.size() == 4) {
                vector<Point2f> src, dst;
                for (int i = 0; i < 4; ++i) src.push_back(Point2f(approx[i].x, approx[i].y));
                sort(src.begin(), src.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
                if (src[0].x > src[1].x) swap(src[0], src[1]);
                if (src[2].x < src[3].x) swap(src[2], src[3]);
                dst = {Point2f(0,0), Point2f((float)warpSize.width-1,0), Point2f((float)warpSize.width-1,(float)warpSize.height-1), Point2f(0,(float)warpSize.height-1)};
                Mat M = getPerspectiveTransform(src, dst);
                warpPerspective(frame, warped, M, warpSize);
                warpPerspective(bin, warpedBin, M, warpSize);
                cvtColor(warped, gray, COLOR_BGR2GRAY);
            }
        }
        vector<Point2f> dots;
        vector<vector<Point>> smallContours;
        findContours(warpedBin.clone(), smallContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& c : smallContours) {
            double area = contourArea(c);
            if (area < 30 || area > 400) continue;
            Rect box = boundingRect(c);
            float aspect = (float)box.width / box.height;
            if (aspect < 0.5f || aspect > 1.5f) continue;
            double peri = arcLength(c, true);
            double circ = (peri > 0) ? 4.0 * CV_PI * area / (peri * peri) : 0;
            if (circ < 0.35) continue;
            Point2f center;
            float radius;
            minEnclosingCircle(c, center, radius);
            dots.push_back(center);
            circle(warped, center, 6, Scalar(0, 0, 255), 2);
        }
        if (dots.size() < 6) {
            Mat blurred;
            medianBlur(gray, blurred, 5);
            vector<Vec3f> circles;
            HoughCircles(blurred, circles, HOUGH_GRADIENT, 1.5, 20, 100, 15, 3, 30);
            for (size_t i = 0; i < circles.size(); ++i) {
                Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                bool exists = false;
                for (auto& d : dots) if (norm(d - center) < 6.0f) { exists = true; break; }
                if (!exists) {
                    dots.push_back(center);
                    circle(warped, center, 6, Scalar(255, 0, 0), 2);
                }
            }
        }
        if (dots.empty()) {
            imshow("Braille", warped);
            imshow("Binary", warpedBin);
            if (waitKey(30) == 27) break;
            continue;
        }
        sort(dots.begin(), dots.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
        vector<float> ydiffs;
        for (size_t i = 1; i < dots.size(); ++i) ydiffs.push_back(abs(dots[i].y - dots[i - 1].y));
        float avgYdiff = 0.f;
        if (!ydiffs.empty()) { float s = 0.f; for (auto v : ydiffs) s += v; avgYdiff = s / ydiffs.size(); }
        const float rowThresh = max(15.f, avgYdiff * 0.7f);
        vector<vector<Point2f>> rows;
        vector<Point2f> row;
        for (auto& p : dots) {
            if (row.empty() || abs(p.y - row.back().y) < rowThresh) row.push_back(p);
            else { rows.push_back(row); row.clear(); row.push_back(p); }
        }
        if (!row.empty()) rows.push_back(row);
        string finalText = "";
        bool numberMode = false;
        bool capitalizeNext = false;
        for (auto& r : rows) {
            sort(r.begin(), r.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
            vector<float> xdiffs;
            for (size_t i = 1; i < r.size(); ++i) xdiffs.push_back(r[i].x - r[i - 1].x);
            float avgXdiff = 0.f;
            if (!xdiffs.empty()) { float s = 0.f; for (auto v : xdiffs) s += v; avgXdiff = s / xdiffs.size(); }
            const float gapEst = max(20.f, avgXdiff * 1.4f);
            vector<vector<Point2f>> cells;
            vector<Point2f> cell;
            for (size_t i = 0; i < r.size(); i++) {
                if (i == 0 || abs(r[i].x - r[i - 1].x) < gapEst) cell.push_back(r[i]);
                else { cells.push_back(cell); cell.clear(); cell.push_back(r[i]); }
            }
            if (!cell.empty()) cells.push_back(cell);
            for (size_t i = 0; i < cells.size(); i++) {
                auto& c = cells[i];
                if (c.empty()) continue;
                int mask = decodeCellMask(c);
                if (mask == BRAILLE_CAPITAL) { capitalizeNext = true; continue; }
                if (mask == BRAILLE_NUMBER) { numberMode = true; continue; }
                char letter = '?';
                if (numberMode && brailleNumbers.count(mask)) {
                    letter = '0' + brailleNumbers[mask];
                } else if (brailleLetters.count(mask)) {
                    letter = brailleLetters[mask];
                    numberMode = false;
                } else if (brailleSymbols.count(mask)) {
                    letter = brailleSymbols[mask];
                }
                if (capitalizeNext) { letter = toupper(letter); capitalizeNext = false; }
                finalText += letter;
                Rect bbox = rectFromPoints(c);
                rectangle(warped, bbox, Scalar(0, 255, 0), 2);
                if (i > 0) {
                    float gap = c[0].x - cells[i - 1][0].x;
                    float cellWidth = max(1.0f, (float)bbox.width);
                    if (gap > 2.0f * cellWidth) finalText += ' ';
                }
            }
        }
        string normFinal = normalizeText(finalText);
        string normTarget = normalizeText(targetPhrase);
        int dist = levenshtein(normFinal, normTarget);
        if (dist <= 6) finalText = targetPhrase;
        if (!finalText.empty() && finalText != lastText) {
            cout << finalText << endl;
            out << finalText << endl;
            lastText = finalText;
        }
        putText(warped, finalText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        imshow("Braille", warped);
        imshow("Binary", warpedBin);
        if (waitKey(30) == 27) break;
    }
    out.close();
    return 0;
}
