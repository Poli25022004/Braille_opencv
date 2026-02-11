#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <deque>
#include <unordered_map>

using namespace cv;
using namespace std;

map<int, char> brailleLetters = {
    {1,'A'},{3,'B'},{9,'C'},{25,'D'},{17,'E'},{11,'F'},{27,'G'},{19,'H'},
    {10,'I'},{26,'J'},{5,'K'},{7,'L'},{13,'M'},{29,'N'},{21,'O'},{15,'P'},
    {31,'Q'},{23,'R'},{14,'S'},{30,'T'},{37,'U'},{39,'V'},{58,'W'},{45,'X'},
    {61,'Y'},{53,'Z'}
};

map<int, int> brailleNumbers = {
    {1,1},{3,2},{9,3},{25,4},{17,5},{11,6},{27,7},{19,8},{10,9},{26,0}
};

map<int, char> brailleSymbols = {
    {4,','},{12,';'},{8,':'},{16,'?'},{32,'!'},{48,'.'}
};

const int BRAILLE_CAPITAL = (1 << 5);
const int BRAILLE_NUMBER = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);

const bool ENABLE_DEBUG = true;
const float SAMPLE_FRACTION = 0.20f;
const float GRID_FRACTION = 0.12f;
const int HAMMING_MAX_ACCEPT = 2;

int buildMask(int mat[3][2]) {
    int mask = 0;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            if (mat[r][c]) mask |= (1 << (c * 3 + r));
    return mask;
}

static int popcount32(int x) {
    return __builtin_popcount((unsigned)x);
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
        minX = min(minX, p.x); minY = min(minY, p.y);
        maxX = max(maxX, p.x); maxY = max(maxY, p.y);
    }
    if (minX == FLT_MAX || minY == FLT_MAX || maxX == -FLT_MAX || maxY == -FLT_MAX) return Rect();
    int ix = max(0, (int)floor(minX));
    int iy = max(0, (int)floor(minY));
    int iw = max(1, (int)ceil(maxX) - ix + 1);
    int ih = max(1, (int)ceil(maxY) - iy + 1);
    return Rect(ix, iy, iw, ih);
}

int findClosestKnownMask(int m) {
    if (m == 0) return 0;
    int best = 0, bestd = 100;
    vector<int> keys;
    for (auto& p : brailleLetters) keys.push_back(p.first);
    for (auto& p : brailleNumbers) keys.push_back(p.first);
    for (auto& p : brailleSymbols) keys.push_back(p.first);
    keys.push_back(BRAILLE_CAPITAL);
    keys.push_back(BRAILLE_NUMBER);
    for (int k : keys) {
        int d = popcount32(m ^ k);
        if (d < bestd) { bestd = d; best = k; }
    }
    if (bestd <= HAMMING_MAX_ACCEPT) return best;
    return 0;
}

int voteMasks(int mask, int gridMask, int ccMask, const Mat& roi) {
    unordered_map<int, int> votes;
    if (mask > 0) votes[mask] += 2;
    if (gridMask > 0) votes[gridMask] += 1;
    if (ccMask > 0) votes[ccMask] += 1;

    int best = 0, bestv = 0;
    for (auto& p : votes)
        if (p.second > bestv) { bestv = p.second; best = p.first; }

    if (bestv == 0) {
        int permissiveMat[3][2] = { 0 };
        int w = max(1, roi.cols / 2);
        int h = max(1, roi.rows / 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 2; ++c) {
                int x0 = c * w, y0 = r * h;
                int x1 = (c == 1 ? roi.cols : x0 + w);
                int y1 = (r == 2 ? roi.rows : y0 + h);
                int area = 0, count = 0;
                for (int yy = y0; yy < y1; ++yy)
                    for (int xx = x0; xx < x1; ++xx) {
                        ++area;
                        if (roi.at<uchar>(yy, xx) > 128) ++count;
                    }
                if (area > 0 && count >= max(1, (int)(area * (GRID_FRACTION * 0.9f))))
                    permissiveMat[r][c] = 1;
            }
        int permMask = buildMask(permissiveMat);
        if (permMask != 0) return permMask;
        int nearest = findClosestKnownMask(mask | gridMask | ccMask);
        return nearest ? nearest : mask | gridMask | ccMask;
    }

    if (!(brailleLetters.count(best) || brailleNumbers.count(best) || brailleSymbols.count(best) || best == BRAILLE_CAPITAL || best == BRAILLE_NUMBER)) {
        int nearest = findClosestKnownMask(best);
        if (nearest) return nearest;
    }
    return best;
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
    if (roi.empty()) return -1;
    if (roi.type() != CV_8U) { Mat tmp; roi.convertTo(tmp, CV_8U); roi = tmp; }

    int minDim = min(bbox.width, bbox.height);
    int sampleRadius = max(2, minDim / 6);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c) {
            float sx_local = (c == 0 ? 0.2f : 0.8f) * (float)roi.cols;
            float sy_local = ((r + 0.5f) / 3.0f) * (float)roi.rows;
            int count = 0, area = 0;
            int rmin = max(0, (int)floor(sy_local - sampleRadius));
            int rmax = min(roi.rows - 1, (int)ceil(sy_local + sampleRadius));
            int cmin = max(0, (int)floor(sx_local - sampleRadius));
            int cmax = min(roi.cols - 1, (int)ceil(sx_local + sampleRadius));
            for (int yy = rmin; yy <= rmax; ++yy)
                for (int xx = cmin; xx <= cmax; ++xx) {
                    float dx = xx - sx_local, dy = yy - sy_local;
                    if (dx * dx + dy * dy <= sampleRadius * sampleRadius) {
                        ++area;
                        if (roi.at<uchar>(yy, xx) > 128) ++count;
                    }
                }
            if (area > 0 && count >= max(1, (int)(area * SAMPLE_FRACTION)))
                mat[r][c] = 1;
        }

    int mask = buildMask(mat);

    int gridMat[3][2] = { 0 };
    int cellW = max(1, roi.cols / 2);
    int cellH = max(1, roi.rows / 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c) {
            int x0 = c * cellW, y0 = r * cellH;
            int x1 = (c == 1) ? roi.cols : x0 + cellW;
            int y1 = (r == 2) ? roi.rows : y0 + cellH;
            int area = 0, count = 0;
            for (int yy = y0; yy < y1; ++yy)
                for (int xx = x0; xx < x1; ++xx) {
                    ++area;
                    if (roi.at<uchar>(yy, xx) > 128) ++count;
                }
            if (area > 0 && count >= max(1, (int)(area * GRID_FRACTION)))
                gridMat[r][c] = 1;
        }

    int gridMask = buildMask(gridMat);

    int ccMat[3][2] = { 0 };
    {
        Mat proc;
        Mat k = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(roi, proc, MORPH_OPEN, k);
        proc = proc > 128;
        Mat labels, stats, centroids;
        int ncomp = connectedComponentsWithStats(proc, labels, stats, centroids, 8, CV_32S);
        for (int i = 1; i < ncomp; ++i) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area<2 || area>(sampleRadius * sampleRadius * 12)) continue;
            double cx = centroids.at<double>(i, 0);
            double cy = centroids.at<double>(i, 1);
            int col = (cx < roi.cols * 0.5) ? 0 : 1;
            int row = (int)floor((cy * 3.0) / roi.rows);
            row = min(2, max(0, row));
            ccMat[row][col] = 1;
        }
    }

    int ccMask = buildMask(ccMat);
    int chosen = voteMasks(mask, gridMask, ccMask, roi);

    if (ENABLE_DEBUG)
        cerr << "decodeCellMaskImage: mask=" << mask << " grid=" << gridMask << " cc=" << ccMask << " -> chosen=" << chosen << endl;

    return chosen;
}
