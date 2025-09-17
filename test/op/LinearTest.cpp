#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <chrono>
#include <sstream>
#include <MNN/AutoTime.hpp>
#include "CommonOpCreator.hpp"

using namespace MNN;
using namespace MNN::Express;

// Forward declare the reference function if it's not in a shared header
static void reference_conv2d(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, int batch, int ic, int oc,
                             int ih, int iw, int kh, int kw, ConvertFP32 functor);

// A simplified reference implementation for Linear (using 1x1 Conv logic)
static void reference_linear(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, std::vector<float>& outputDataSeparateBias, int batch, int ic, int oc,
                             ConvertFP32 functor) {
    output.resize(batch * oc);
    outputDataSeparateBias.resize(batch * oc);

    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            float sum = 0.0f;
            auto destOffset = b * oc + oz;
            for (int sz = 0; sz < ic; ++sz) {
                float xValue = input[b * ic + sz];
                float convertX = functor(xValue);
                // Weight layout is [oc, ic]
                float convertW = functor(weight[oz * ic + sz]);
                sum += convertX * convertW;
            }
            output[destOffset] = functor(sum + functor(bias[oz]));
            outputDataSeparateBias[destOffset] = functor(functor(sum) + functor(bias[oz]));
        }
    }
}



class LinearTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Default configuration
        int N = 128; // Input Features
        int C = 128; // Output Features
        const char* flag = "";
        
        // This approach avoids modifying MNNTestSuite.h
        // It assumes g_argc and g_argv are available from main.cpp
        extern int g_argc;
        extern char** g_argv;
        if (g_argc > 5) {
            flag = g_argv[5];
        }

        // Parse configuration from flag
        // format: shape=1024,4096
        std::stringstream ss(flag);
        std::string item;
        while (std::getline(ss, item, ';')) {
            std::stringstream item_ss(item);
            std::string key, value;
            if (std::getline(item_ss, key, '=') && std::getline(item_ss, value)) {
                if (key == "shape") {
                    std::stringstream shape_ss(value);
                    std::string dim;
                    if (std::getline(shape_ss, dim, ',')) {
                        N = std::stoi(dim);
                    }
                    if (std::getline(shape_ss, dim, ',')) {
                        C = std::stoi(dim);
                    }
                }
            }
        }
        
        MNN_PRINT("Running LinearTest with shape: ic=%d, oc=%d\n", N, C);

        bool res = true;
        MNN_PRINT("--- Testing Per-Channel Quantization ---\n");
        // Test for 8 bits - Per-Channel
        res = res && testUnit( "Linear", 1, N, C, precision, 8, true, 0);
        res = res && testUnit("Linear", 1, N, C, precision, 8, false, 0);
        // Test for 4 bits - Per-Channel
        res = res && testUnit( "Linear", 1, N, C, precision, 4, true, 0);
        res = res && testUnit( "Linear", 1, N, C, precision, 4, false, 0);

        if (res) {
            MNN_PRINT("Linear Per-Channel passed!\n");
        } else {
            MNN_PRINT("Linear Per-Channel failed!\n");
        }
 
        MNN_PRINT("--- Testing Per-Block Quantization (blockSize=64) ---\n");
        int blockSize = 64;
        if (N < blockSize) {
            MNN_PRINT("Skipping Per-Block test as ic < blockSize.\n");
        } else {
            // Test for 8 bits - Per-Block
            res = res && testUnit( "Linear", 1, N, C, precision, 8, true, blockSize);
            res = res && testUnit("Linear", 1, N, C, precision, 8, false, blockSize);
            // Test for 4 bits - Per-Block
            res = res && testUnit( "Linear", 1, N, C, precision, 4, true, blockSize);
            res = res && testUnit( "Linear", 1, N, C, precision, 4, false, blockSize);
        }

        if (res) {
            MNN_PRINT("Linear Per-Block passed!\n");
        } else {
            MNN_PRINT("Linear Per-Block failed!\n");
        }
        return res;

    }

private:
    void generateWeight(std::vector<float>& weightData, int ic, int oc) {
        auto numbers = ic * oc;
        weightData.resize(numbers);
        float rate = 1.0f / numbers;
        for (int i = 0; i < numbers; ++i) {
            int data = i - numbers / 2;
            weightData[i] = (float)data * rate;
        }
    }

    bool testUnit(const std::string &test_op_name, int batch, int ic, int oc,
                  int precision, int nbit, bool async, int blockSize) {

        using namespace MNN::Express;
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};

        std::vector<float> weightData, biasData;
        generateWeight(weightData, ic, oc);

        for (int i = 0; i < oc; i++) {
            auto data = (((i % 1317) * (i % 1317)) + i / ic + i / oc +
                        (oc - i) * ic + i * (oc - i)) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            biasData.push_back(floatData);
        }

        std::vector<float> inputData, outputData, outputDataSeparateBias;
        float rate = 1.0f;
        if (ic * batch > 10000) {
            // Avoid exceed fp16 limit
            rate = 0.01f;
        }
        for (int i = 0; i < ic * batch; ++i) {
            auto data = (((i / ic) % 1317) * ((i / oc) % 1317) +
                        ((oc - i) % 1317) * ic + (i % 1317) * ((oc - i) % 1317));
            data = (data * data) % 1317;
            auto floatData = (float)(data % 255) / 255.0f * rate;
            inputData.push_back(floatData);
        }

        // Quantize weight and get dequantized weight for reference
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold;
        if (async) {
            clampMin = -threshold - 1;
        }
        
        int kernel_size = ic;
        int num_blocks = 1;

        if (blockSize == 0) 
            blockSize = kernel_size; // Per-Channel
        else 
            num_blocks = (kernel_size + blockSize - 1) / blockSize;

        std::vector<int8_t> quantWeight(oc * ic);
        std::vector<float> dequantizedWeightData = weightData;
        std::vector<float> wScale;

        if (async) {
            // ASYMMETRIC PER-BLOCK QUANTIZATION
            wScale.resize(2 * oc * num_blocks);
            for (int k = 0; k < oc; ++k) { // Loop over output channels
                for (int g = 0; g < num_blocks; ++g) { // Loop over blocks within the channel
                    int currentBlockSize = blockSize;
                    int beginIndex = k * kernel_size + g * blockSize;
                    if (g == num_blocks - 1) { // Handle the last block which might be smaller
                        currentBlockSize = kernel_size - g * blockSize;
                    }

                    auto minMax = findMinMax(dequantizedWeightData.data() + beginIndex, currentBlockSize);
                    auto minValue = minMax.first;
                    auto absMax = minMax.second - minMax.first;
                    
                    int scaleIndex = (k * num_blocks + g) * 2;
                    wScale[scaleIndex] = minValue;
                    wScale[scaleIndex + 1] = 0;
                    
                    float quantscale = 1.0f;
                    if (absMax >= 1e-6f) {
                        wScale[scaleIndex + 1] = absMax / (threshold - clampMin);
                        quantscale = 1.0f / wScale[scaleIndex + 1];
                    }

                    float* ptr = dequantizedWeightData.data() + beginIndex;
                    for (int i = 0; i < currentBlockSize; ++i) {
                        int8_t quantValue = (int8_t)roundf((ptr[i] - minValue) * quantscale + clampMin);
                        quantWeight[beginIndex + i] = quantValue;
                        ptr[i] = ((float)quantValue - clampMin) * wScale[scaleIndex + 1] + minValue;
                    }
                }
            }
        } else {
            // SYMMETRIC PER-BLOCK QUANTIZATION
            wScale.resize(oc * num_blocks);
            for (int k = 0; k < oc; ++k) { // Loop over output channels
                for (int g = 0; g < num_blocks; ++g) { // Loop over blocks within the channel
                    int currentBlockSize = blockSize;
                    int beginIndex = k * kernel_size + g * blockSize;
                    if (g == num_blocks - 1) { // Handle the last block
                        currentBlockSize = kernel_size - g * blockSize;
                    }

                    auto absMax = findAbsMax(dequantizedWeightData.data() + beginIndex, currentBlockSize);
                    int scaleIndex = k * num_blocks + g;
                    wScale[scaleIndex] = absMax / threshold;
                    if (absMax < 1e-6f) { // Avoid division by zero
                        wScale[scaleIndex] = 0.0f;
                    }

                    float* ptr = dequantizedWeightData.data() + beginIndex;
                    for (int i = 0; i < currentBlockSize; ++i) {
                        float scale = wScale[scaleIndex] == 0.0f ? 0.0f : (ptr[i] / wScale[scaleIndex]);
                        int8_t quantVal = (int8_t)(fmaxf(fminf(roundf(scale), threshold), clampMin));
                        quantWeight[beginIndex + i] = quantVal;
                        ptr[i] = (float)quantVal * wScale[scaleIndex];
                    }
                }
            }
        }

        reference_linear(inputData, dequantizedWeightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, FP32Converter[precision]);

        // Use 1x1 Conv to simulate Linear
        auto input = _Input({batch, ic, 1, 1}, NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

        auto weightLength = weightData.size();
        float errorScale = 1.0f;
        if (nbit == 4 && weightLength > 10000) {
            errorScale = 50.0f;
        }
        int memory = MNNTestSuite::get()->pStaus.memory;
        if (precision > MNN::BackendConfig::Precision_High || memory > MNN::BackendConfig::Memory_High) {
            errorScale = 100.0f;
        }

        auto output = _HybridConv(weightData, biasData, wScale, input,
                                {ic, oc}, {1, 1}, VALID, {1, 1}, {1, 1}, 1, {0, 0}, false, false, nbit, async);
       
        output = _Convert(output, NCHW);
        auto outputPtr = output->readMap<float>();



        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.01 * errorScale)) {
            MNN_ERROR("%s test failed for %d bits, async=%d, shape=[%d, %d]!\n",
                    test_op_name.c_str(), nbit, async, ic, oc);
            return false;
        }
        
        return true;
    }
};

MNNTestSuiteRegister(LinearTest, "op/linear");