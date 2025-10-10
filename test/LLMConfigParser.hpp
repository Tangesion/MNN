#ifndef LLM_CONFIG_PARSER_Hpp
#define LLM_CONFIG_PARSER_Hpp

#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"
#include <string>
#include <iostream>
#include <cstdio>
#include <utility>

class LLMConfigParser {
public:
    // Public struct to hold the configuration data for easy access
    struct ModelConfig {
        std::string model_type;
        int hidden_size;
        int num_attention_heads;
        int num_key_value_heads;
        int head_dim;
        int vocab_size;
        int intermediate_size;
        
        // Initialize with default values
        ModelConfig() : hidden_size(0), num_attention_heads(0), num_key_value_heads(0), head_dim(0), vocab_size(0), intermediate_size(0){}
    };

    struct LinearShape {
        std::pair<int, int> qProj;
        std::pair<int, int> kvProj;
        std::pair<int, int> oProj;
        std::pair<int, int> gateProj;
        std::pair<int, int> upProj;
        std::pair<int, int> downProj;
        std::pair<int, int> lmHead;
    };

    // Public member to store the parsed config
    ModelConfig config;
    LinearShape shape;

    /**
     * @brief Default constructor. Creates an empty parser.
     */
    LLMConfigParser() : loaded_(false) {}

    /**
     * @brief Constructs the parser and immediately parses the JSON file at the given path.
     * @param path The path to the config.json file.
     */
    LLMConfigParser(const std::string& path) : loaded_(false) {
        parse(path);
    }

    /**
     * @brief Checks if the config file was successfully loaded and parsed.
     * @return True if successful, false otherwise.
     */
    bool IsLoaded() const {
        return loaded_;
    }
    

private:
    // The handler is an implementation detail, so we make it a private nested class.
    struct ConfigHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ConfigHandler> {
        ModelConfig& config;
        std::string currentKey;

        ConfigHandler(ModelConfig& cfg) : config(cfg) {}

        bool Key(const char* str, rapidjson::SizeType length, bool copy) {
            currentKey.assign(str, length);
            return true;
        }

        bool String(const char* str, rapidjson::SizeType length, bool copy) {
            if (currentKey == "model_type") {
                config.model_type.assign(str, length);
            }
            return true;
        }

        bool Int(int i) {
            if (currentKey == "hidden_size") {
                config.hidden_size = i;
            } else if (currentKey == "num_attention_heads") {
                config.num_attention_heads = i;
            } else if (currentKey == "num_key_value_heads") {
                config.num_key_value_heads = i;
            } else if (currentKey == "head_dim") {
                config.head_dim = i;
            } else if (currentKey == "vocab_size") {
                config.vocab_size = i;
            } else if (currentKey == "intermediate_size") {
                config.intermediate_size = i;
            }
            return true;
        }

        // Default handlers for types we don't care about
        bool Null() { return true; }
        bool Bool(bool b) { return true; }
        bool Uint(unsigned u) { return Int(u); }
        bool Int64(int64_t i) { return true; }
        bool Uint64(uint64_t u) { return true; }
        bool Double(double d) { return true; }
        bool StartObject() { return true; }
        bool EndObject(rapidjson::SizeType mc) { return true; }
        bool StartArray() { return true; }
        bool EndArray(rapidjson::SizeType ec) { return true; }
    };

    // Private method to handle the parsing logic
    void parse(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            return;
        }

        char readBuffer[65536];
        rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

        ConfigHandler handler(config); // The handler will populate our 'config' member
        rapidjson::Reader reader;
        rapidjson::ParseResult result = reader.Parse(is, handler);
        
        fclose(fp);

        if (!result) {
            std::cerr << "JSON parse error at offset " << result.Offset() << ": "
                      << result.Code() << std::endl; // Simplified error message
            return;
        }
        
        shape.qProj = {config.hidden_size, config.num_attention_heads * config.head_dim};
        shape.kvProj = {config.hidden_size, config.num_key_value_heads * config.head_dim};
        shape.oProj = {config.num_attention_heads * config.head_dim, config.hidden_size};
        shape.gateProj = {config.hidden_size, config.intermediate_size};
        shape.upProj = {config.hidden_size, config.intermediate_size};
        shape.downProj = {config.intermediate_size, config.hidden_size};
        shape.lmHead = {config.hidden_size, config.vocab_size};

        loaded_ = true; // Set flag to true on successful parse
        
        
    }


    bool loaded_; // Flag to indicate if parsing was successful
};



#endif