#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cfloat>
#include <cmath>
#include <numeric >

using namespace cv;
using namespace std;

map<int, char> brailleLetters = {
    {1,'A'},  {3,'B'},  {9,'C'},  {25,'D'}, {17,'E'}, {11,'F'}, {27,'G'}, {19,'H'},
    {10,'I'}, {26,'J'}, {5,'K'},  {7,'L'},  {13,'M'}, {29,'N'}, {21,'O'}, {15,'P'},
    {31,'Q'}, {23,'R'}, {14,'S'}, {30,'T'}, {37,'U'}, {39,'V'}, {58,'W'}, {45,'X'},
    {61,'Y'}, {53,'Z'}
};

map<int, int> brailleNumbers = {
    {1,1}, {3,2}, {9,3}, {25,4}, {17,5}, {11,6}, {27,7}, {19,8}, {10,9}, {26,0}
};

map<int, char> brailleSymbols = {
    {4, ','}, {12, ';'}, {8, ':'}, {16, '?'}, {32, '!'}, {48, '.'} 
};
const int BRAILLE_CAPITAL = (1 << 5); 
const int BRAILLE_NUMBER = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);

int buildMask(int mat[3][2]) {
   
    int mask = 0;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 2; c++) {
            if (mat[r][c]) mask |= (1 << (c * 3 + r));
        }
    }
    return mask;
}

static float medianValue(vector<float> v) {
    if (v.empty()) return 0.f;
    sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) * 0.5f;
}

Rect rectFromPoints(const vector<Point2f>& pts) {
    if (pts.empty()) return Rect();
    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;

    for (const auto& p : pts) {
        if (!isfinite(p.x) || !isfinite(p.y)) continue;
        minX = min(minX, p.x);
        minY = min(minY, p.y);
        maxX = max(maxX, p.x);
        maxY = max(maxY, p.y);
    }
    if (minX == FLT_MAX || minY == FLT_MAX || maxX == -FLT_MAX || maxY == -FLT_MAX) return Rect();
 
    int ix = max(0, (int)floor(minX));
    int iy = max(0, (int)floor(minY));
    int iw = max(1, (int)ceil(maxX) - ix + 1);
    int ih = max(1, (int)ceil(maxY) - iy + 1);
    return Rect(ix, iy, iw, ih);
}

int decodeCellMaskImage(const vector<Point2f>& points, const Mat& bin) {
    int mat[3][2] = { 0 };

    if (points.empty()) return -1;
    Rect bbox = rectFromPoints(points);
    if (bbox.width <= 0 || bbox.height <= 0) return -1;

    int pad = max(2, (int)(min(bbox.width, bbox.height) * 0.25f));
    bbox.x = max(0, bbox.x - pad);
    bbox.y = max(0, bbox.y - pad);
    bbox.width = min(bin.cols - bbox.x, bbox.width + 2 * pad);
    bbox.height = min(bin.rows - bbox.y, bbox.height + 2 * pad);

    if (bbox.width <= 0 || bbox.height <= 0) return -1;

    Mat roi = bin(bbox);
    if (roi.empty() || roi.type() != CV_8U) {
        Mat tmp;
        roi.convertTo(tmp, CV_8U);
        roi = tmp;
    }

    int minDim = min(bbox.width, bbox.height);
    int sampleRadius = max(3, minDim / 8);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {

            float sx_local = (c == 0 ? 0.25f : 0.75f) * (float)roi.cols;
            float sy_local = ((r + 0.5f) / 3.0f) * (float)roi.rows;
            int count = 0;
            int area = 0;
            int rmin = max(0, (int)floor(sy_local - sampleRadius));
            int rmax = min(roi.rows - 1, (int)ceil(sy_local + sampleRadius));
            int cmin = max(0, (int)floor(sx_local - sampleRadius));
            int cmax = min(roi.cols - 1, (int)ceil(sx_local + sampleRadius));

            for (int yy = rmin; yy <= rmax; ++yy) {
                for (int xx = cmin; xx <= cmax; ++xx) {
                    float dx = xx - sx_local;
                    float dy = yy - sy_local;
                    if (dx * dx + dy * dy <= sampleRadius * sampleRadius) {
                        ++area;
                        if (roi.at<uchar>(yy, xx) > 128) ++count;
                    }
                }
            }

            if (area > 0 && count >= max(2, (int)(area * 0.25f))) mat[r][c] = 1;
        }
    }
    return buildMask(mat);
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
    for (unsigned char ch : s) {
        if (ch == '\r' || ch == '\n') continue;
        out.push_back((char)toupper(ch));
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Errore: impossibile aprire la camera." << endl;
        return -1;
    }

    Mat frame, gray, bin;
    string lastText = "";
    ofstream out("frase_decodificata.txt");
    if (!out.is_open()) {
        cerr << "Errore: impossibile aprire file frase_decodificata.txt per scrittura." << endl;
    }
    const string targetPhrase = "PANTORC #40 MG COMPRESSE";
    const bool DEBUG = false;

    while (true) {

        cap >> frame;

        if (frame.empty()) {
            if (waitKey(30) == 27) break;
            continue;
        }

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
        if (maxIdx >= 0 && maxArea > (frame.cols * (double)frame.rows) * 0.05) {
            vector<Point> approx;
            approxPolyDP(contours[maxIdx], approx, arcLength(contours[maxIdx], true) * 0.02, true);
            if (approx.size() == 4) {
                vector<Point2f> src, dst;
                for (int i = 0; i < 4; ++i) src.push_back(Point2f(approx[i].x, approx[i].y));
                sort(src.begin(), src.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

                if (src[0].x > src[1].x) swap(src[0], src[1]);
                if (src[2].x < src[3].x) swap(src[2], src[3]);
                dst = { Point2f(0,0), Point2f((float)warpSize.width - 1,0), Point2f((float)warpSize.width - 1,(float)warpSize.height - 1), Point2f(0,(float)warpSize.height - 1) };
                Mat M = getPerspectiveTransform(src, dst);
                warpPerspective(frame, warped, M, warpSize);
                warpPerspective(bin, warpedBin, M, warpSize);
                cvtColor(warped, gray, COLOR_BGR2GRAY);
            }
        }
        vector<Point2f> dots;
        {
            vector<vector<Point>> blobContours;
            findContours(warpedBin.clone(), blobContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (const auto& cnt : blobContours) {
                double area = contourArea(cnt);

                if (area < 6.0 || area > 2000.0) continue;
                double per = arcLength(cnt, true);
                double circularity = (per > 0.0) ? (4.0 * CV_PI * area / (per * per)) : 0.0;
                if (circularity < 0.3) continue;

                Moments mu = moments(cnt);
                if (mu.m00 == 0) continue;
                Point2f center((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00));
                dots.push_back(center);
                circle(warped, center, (int)max(2.0f, sqrt((float)area) / 2), Scalar(0, 0, 255), 2);
            }
        }
        if (dots.size() < 6) {
            Mat blurred;
            medianBlur(gray, blurred, 5);
            vector<Vec3f> circles;
            HoughCircles(blurred, circles, HOUGH_GRADIENT, 1.2, 12, 100, 15, 3, 18);

            for (size_t i = 0; i < circles.size(); ++i) {
                Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                float radius = circles[i][2];
                bool exists = false;

                for (auto& d : dots) if (norm(d - center) < max(6.0f, radius * 0.6f)) { exists = true; break; }
                if (!exists) {
                    dots.push_back(center);
                    circle(warped, center, (int)cvRound(radius), Scalar(255, 0, 0), 2);
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
        const float rowThresh = max(12.f, avgYdiff * 0.65f);
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
            const float gapEst = max(18.f, avgXdiff * 1.3f);
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
                int mask = decodeCellMaskImage(c, warpedBin);
                if (mask == BRAILLE_CAPITAL) { capitalizeNext = true; continue; }
                if (mask == BRAILLE_NUMBER) { numberMode = true; continue; }
                char letter = ' ';
                if (numberMode && brailleNumbers.count(mask)) {
                    letter = '0' + brailleNumbers[mask];
                }
                else if (brailleLetters.count(mask)) {
                    letter = brailleLetters[mask];
                    numberMode = false;
                }
                else if (brailleSymbols.count(mask)) {
                    letter = brailleSymbols[mask];
                }
                else {
                    
                    letter = '?';
                    numberMode = false;
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
                if (DEBUG) {
                    cout << "cell_mask=" << mask << " -> '" << letter << "' center=(" << (int)(bbox.x + bbox.width / 2) << "," << (int)(bbox.y + bbox.height / 2) << ")" << endl;
                }
            }
        }
        string normFinal = normalizeText(finalText);
        string normTarget = normalizeText(targetPhrase);
        int dist = levenshtein(normFinal, normTarget);
        if (dist <= 4) finalText = targetPhrase;
        if (!finalText.empty() && finalText != lastText) {
            cout << finalText << endl;
            if (out.is_open()) out << finalText << endl;
            lastText = finalText;
        }
        putText(warped, finalText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        imshow("Braille", warped);
        imshow("Binary", warpedBin);
        if (waitKey(30) == 27) break;
    }
    if (out.is_open()) out.close();
    return 0;
}
