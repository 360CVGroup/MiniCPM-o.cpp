#ifndef EXAMPLE_VIDEO_DECODER_H_
#define EXAMPLE_VIDEO_DECODER_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include "utils.h"

#ifndef SWR_CH_MAX
#define SWR_CH_MAX 32
#endif

#define AVCODEC_FREE(avcodec_ctx)               \
    do {                                        \
        if (avcodec_ctx != nullptr) {           \
            avcodec_free_context(&avcodec_ctx); \
            avcodec_ctx = nullptr;              \
        }                                       \
    } while (0);

#define AVFRAME_FREE(frame)        \
    do {                           \
        if (frame != nullptr) {    \
            av_frame_free(&frame); \
            frame = nullptr;       \
        }                          \
    } while (0);

namespace edge {
class VideoDecoder {
public:
    VideoDecoder() = delete;
    VideoDecoder(std::string video_path);

    void decode(double sec_interval = 1.0);

    std::vector<float> get_audio_pcmf32();
    std::vector<image_buf<uint8_t>> get_video_buffer();

    void write_wav(std::string file_name);
    void write_jpg(std::string prefix_name);

    ~VideoDecoder() {
        // free buffer
        if (converted_data_buf_[0]) {
            av_freep(&converted_data_buf_[0]);
        }

        if (fmt_ctx_ != nullptr) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
        }

        AVCODEC_FREE(video_codec_ctx_);
        AVCODEC_FREE(audio_codec_ctx_);

        AVFRAME_FREE(raw_frame_);
        AVFRAME_FREE(rgb_frame_);
        AVFRAME_FREE(audio_frame_);

        if (sws_ctx_ != nullptr) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (swr_ctx_ != nullptr) {
            swr_free(&swr_ctx_);
            swr_ctx_ = nullptr;
        }
        if (fmt_ctx_ != nullptr) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
        }
    }

protected:
    int init_video_decoder();
#ifndef MACOS_STREAM
    int init_audio_decoder();
#else
    int init_audio_decoder_macos();
#endif

private:
    int image_max_length_ = 640;
    int wav_sample_rate_  = 16000;
    int out_nb_channels_  = 1;  // single channel

    uint8_t* converted_data_buf_[SWR_CH_MAX] = {nullptr};

    std::vector<image_buf<uint8_t>> video_buffers_;
    std::vector<float> pcmf32_data_;

    int video_stream_idx_ = -1;
    int audio_stream_idx_ = -1;

    AVFormatContext* fmt_ctx_ = nullptr;

    SwsContext* sws_ctx_ = nullptr;
    SwrContext* swr_ctx_ = nullptr;

    AVCodecContext* video_codec_ctx_ = nullptr;
    AVCodecContext* audio_codec_ctx_ = nullptr;

    AVFrame* raw_frame_ = nullptr;
    AVFrame* rgb_frame_ = nullptr;

    AVFrame* audio_frame_ = nullptr;
};
}  // namespace edge

#endif  // EXAMPLE_VIDEO_DECODER_H_
