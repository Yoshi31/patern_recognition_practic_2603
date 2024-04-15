#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    VideoCapture videoInput("C:/Users/zayka/Source/Repos/CV_NGTU/VID_20240326_103227.mp4");

    if (!videoInput.isOpened()) {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    int frameWidth = static_cast<int>(videoInput.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(videoInput.get(CAP_PROP_FRAME_HEIGHT));

    namedWindow("Original Video", WINDOW_NORMAL);
    resizeWindow("Original Video", frameWidth, frameHeight);

    namedWindow("HSV Mask", WINDOW_NORMAL);
    resizeWindow("HSV Mask", frameWidth, frameHeight);

    // HSV threshold values
    int lowH = 95, lowS = 12, lowV = 138;
    int highH = 255, highS = 50, highV = 255;

    // Create trackbars for HSV thresholds
    createTrackbar("Low H", "HSV Mask", &lowH, 255);
    createTrackbar("High H", "HSV Mask", &highH, 255);
    createTrackbar("Low S", "HSV Mask", &lowS, 255);
    createTrackbar("High S", "HSV Mask", &highS, 255);
    createTrackbar("Low V", "HSV Mask", &lowV, 255);
    createTrackbar("High V", "HSV Mask", &highV, 255);

    VideoWriter videoOutput("output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, Size(frameWidth, frameHeight));

    while (true) {
        Mat frame, hsv, mask, edges;
        videoInput >> frame;

        if (frame.empty())
            break;

        // Preprocessing
        GaussianBlur(frame, frame, Size(5, 5), 0);
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Scalar lowerThreshold = Scalar(lowH, lowS, lowV);
        Scalar upperThreshold = Scalar(highH, highS, highV);

        // Creating mask using HSV thresholds
        inRange(hsv, lowerThreshold, upperThreshold, mask);
        Canny(mask, edges, 30, 90);

        // Morphological operations
        dilate(edges, edges, Mat(), Point(-1, -1), 2);
        erode(edges, edges, Mat(), Point(-1, -1), 2);

        // Contour detection
        std::vector<std::vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = contourArea(contour);

            if (area > 1000) {
                RotatedRect rotatedRect = minAreaRect(contour);
                Point2f vertices[4];
                rotatedRect.points(vertices);

                for (int j = 0; j < 4; ++j) {
                    line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2);
                }
            }
        }

        imshow("Original Video", frame);
        imshow("HSV Mask", mask);

        videoOutput.write(frame);

        if (waitKey(33) == 'q')
            break;
    }

    videoInput.release();
    destroyAllWindows();
    videoOutput.release();

    return 0;
}