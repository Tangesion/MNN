#include <cstdint>
#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNNTestSuite.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Session.hpp"
#include <regex>
#include <thread>
#include <vector>
#include "MNN_generated.h"

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
        res = res && testLinear("Linear Asymm Block64 4bit", 1, 1024, 1024, 4, true, 64);
        return res;
    }

private:
    bool testLinear(std::string title, int batch, int ic, int oc, int nbit = 8, bool asymmetric = false, int blockSize = 0) {
        std::vector<int> bias(oc);
        int numBlocks = (blockSize > 0) ? ic / blockSize : 1;
        int scaleSize = asymmetric ? oc * numBlocks * 2 : oc * numBlocks;
        std::vector<int> channel = {ic, oc};
        std::vector<int> kernel = {1, 1};
        std::vector<int> strides = {1, 1}, dilate = {1, 1}, pad = {1, 1};
        std::vector<float> scale(scaleSize);
        std::vector<int8_t> weight(oc * ic);
        VARP x = _Input({batch, ic, 1, 1}, MNN::Express::NC4HW4, halide_type_of<int8_t>());

        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<int8_t>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;

        for (int i = 0; i < oc; ++i) {
            bias[i] = (10000 + i*i*10 - i*i*i) % 12580;
            for (int j = 0; j < ic; ++j) {
                weight[i * ic + j] = (i * 13 + j * 37 + 1234) % (xMax - xMin + 1) + xMin;
            }
        }

        for (int i = 0; i < int(scaleSize / 2); ++i) {
            scale[i * 2] =  2;
            scale[i * 2 + 1] = fabs(((127-i)*i % 128) / 20000.0f);
        }
        x = _FloatToInt8(_Cast<float>(x), _Scalar<float>(1.0f), -127, 127);
        auto y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), x,
                           channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, 0, 0, -127, 127, false);

        if (nbit != 8) {
            std::unique_ptr<MNN::OpT> op(y->expr().first->get()->UnPack());
            op->main.AsConvolution2D()->symmetricQuan->nbits = nbit;
            y = Variable::create(Expr::create(op.get(), {x}));
            op.reset();
        }
        auto yr = _Int8ToFloat(y, _Scalar<float>(1.0f));
        yr = _Cast<int8_t>(yr);
        auto yInfo = y->getInfo();
        auto yPtr  = yr->readMap<int8_t>();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];

        {
            x.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<float>();
                y->readMap<float>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("%s kernel=(%dx%d) input=(1x%dx%dx%d) output=(1x%dx%dx%d) stride=(%dx%d), avg time = %f\n",
                      title.c_str(), 1, 1, ic, 1, 1, oc, oh, ow, strides[1], strides[0], 1.0 * time / LOOP);
        }

        return true;

    }
};

MNNTestSuiteRegister(LinearSpeedTest, "speed/LinearInt8");