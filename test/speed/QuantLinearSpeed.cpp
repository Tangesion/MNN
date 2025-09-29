
#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include "CommonOpCreator.hpp"

using namespace MNN::Express;
using namespace MNN;

class LinearSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        bool res = true;
        // Symmetric, Per-channel
        res = res && testLinear("Linear Symm", 1, 1024, 1024, 8, false, 0);
        // Asymmetric, Per-channel
        res = res && testLinear("Linear Asymm", 1, 1024, 1024, 8, true, 0);
        // Asymmetric, Block-wise
        res = res && testLinear("Linear Asymm Block64", 1, 1024, 1024, 8, true, 64);
        // 4-bit version
        res = res &&
            testLinear("Linear Asymm Block64 4bit", 1, 1024, 1024, 4, true, 64);
        return res;
  }

private:
    bool testLinear(std::string title, int batch, int ic, int oc, int nbit = 8,
                  bool asymmetric = false, int blockSize = 0) {
        std::vector<int> kernel = {1, 1};
        std::vector<int> pad = {0, 0};
        std::vector<int> channel = {ic, oc};
        std::vector<int> stride = {1, 1};
        std::vector<int> dilate = {1, 1};

        float fac = 0.23;
        int res = 10;
        float tail = 0.05;
        int iw = 1, ih = 1;
        std::vector<float> bias(oc), biastest(oc), biasdup(oc);
        int blockNum = 1;
        if (0 == blockSize || ic % blockSize != 0) {
            blockSize = ic;
            blockNum = 1;
        } else {
            blockNum = ic / blockSize;
        }

        std::vector<float> weightFp32(oc * ic);

        std::vector<float> wScale;
        if (asymmetric) {
            wScale.resize(2 * oc * blockNum);
        } else {
            wScale.resize(oc * blockNum);
        }

        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold - 1;
        VARP x = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<float>();
        int8_t xMin = -(1 << (nbit - 1)), xMax = (1 << (nbit - 1)) - 1;
        for (int i = 0; i < xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1) - (xMax / 2)) * 0.017;
        }
        x = _Convert(x, NC4HW4);
        for (int i = 0; i < oc; ++i) {
            bias[i] = i % 10 + 0.005;
            for (int j = 0; j < ic; ++j) {
                weightFp32[i * ic + j] = (i * ic + j) % res * fac + tail;
            }
        }
        ::memcpy(biastest.data(), bias.data(), oc * sizeof(float));
        ::memcpy(biasdup.data(), bias.data(), oc * sizeof(float));
        int kernel_size = ic;
        auto newWeightFp32 = weightFp32;

        for (int k = 0; k < oc; ++k) {
            for (int j = 0; j < blockNum; ++j) {
                int index = k * blockNum + j;
                auto minmax = findMinMax(weightFp32.data() + k * ic + j * blockSize, blockSize);
                if (asymmetric) {
                    auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);
                    wScale[2 * index] = minmax.first;
                    wScale[2 * index + 1] = scale_;
                    for (int u = 0; u < blockSize; ++u) {
                        int idx = k * ic + j * blockSize + u;
                        int q_weight = (weightFp32[idx] - minmax.first) * (threshold - clampMin) / (minmax.second - minmax.first) + clampMin;
                        newWeightFp32[idx] = (q_weight - xMin) * scale_ + minmax.first;
                    }
                } else {
                    float absMax = std::max(std::abs(minmax.first), std::abs(minmax.second));
                    auto scale_ = absMax / threshold;
                    wScale[index] = scale_;
                    for (int u = 0; u < blockSize; ++u) {
                        int idx = k * ic + j * blockSize + u;
                        int q_weight = weightFp32[idx] / scale_;
                        newWeightFp32[idx] = q_weight * scale_;
                    }
                }
            }
        }

        auto y     = _HybridConv(weightFp32, std::move(bias), std::move(wScale), x, channel, kernel, PaddingMode::CAFFE, stride, dilate, 1, pad, false, false, nbit, asymmetric);
        auto yfp32 = _Conv(std::move(newWeightFp32), std::move(biasdup), x, {ic, oc}, kernel, PaddingMode::CAFFE, stride, dilate, 1, pad);
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
#if defined (__aarch64__) && (precision == 2)
#define FLOAT_T __fp16
#else
#define FLOAT_T float
#endif
        y = _Convert(y, NCHW);
        yfp32 = _Convert(yfp32, NCHW);
        auto yPtr  = y->readMap<FLOAT_T>();
        auto tgPtr = yfp32->readMap<FLOAT_T>();
        auto elesize = yfp32->getInfo()->size;
        float limit = 0.1f;
        bool correct = true;
        float maxValue = 0.001f;
        for (int i = 0; i < elesize; ++i) {
            maxValue = fmaxf(maxValue, fabsf(tgPtr[i]));
        }

        for (int i = 0; i < elesize; ++i) {
            float targetValue = tgPtr[i], computeResult = yPtr[i];
            float diff = targetValue - computeResult;
            float ratio = fabsf(diff) / maxValue;
            if (ratio > limit) {
                MNN_PRINT("%d result Error ratio=%f: right=%f, error=%f\n", i, ratio, targetValue, computeResult);
                MNN_PRINT("conv info: input=(%dx%dx%dx%d) output=(%dx%dx%dx%d)\n", batch, ic, ih, iw, batch, oc, oh, ow);
                correct = false;
                break;
            }
        }
        x.fix(VARP::INPUT);
        const int LOOP = 100;
        {
            x->writeMap<FLOAT_T>();
            y->readMap<FLOAT_T>();
        }
        MNN::Timer _t;
        for (int i = 0; i < LOOP; ++i) {
            x->writeMap<FLOAT_T>();
            y->readMap<FLOAT_T>();
        }
        auto time = (float)_t.durationInUs() / 1000.0f;
        MNN_PRINT("%s input=(%dx%dx%dx%d) output=(%dx%dx%dx%d) avg time = %f\n",
                    title.c_str(), batch, ic, ih, iw, batch, oc, oh, ow, 1.0 * time / LOOP);
        
        return correct;

        
    }
};

MNNTestSuiteRegister(LinearSpeedTest, "speed/LinearInt8");