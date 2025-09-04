#ifndef EXAMPLE_COMMON_H_
#define EXAMPLE_COMMON_H_

#include <string>
#include <vector>

// only support 16k sample rate
#define COMMON_SAMPLE_RATE 16000

namespace edge {
bool is_wav_buffer(const std::string buf);
bool read_wav(const std::string& fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo);
}  // namespace edge

#endif  // EXAMPLE_COMMON_H_
