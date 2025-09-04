#include "video_decoder.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"

namespace edge {

#ifndef MACOS_STREAM
int VideoDecoder::init_audio_decoder() {
    // init audio decoder
    audio_codec_ctx_ = avcodec_alloc_context3(avcodec_find_decoder(fmt_ctx_->streams[audio_stream_idx_]->codecpar->codec_id));
    avcodec_parameters_to_context(audio_codec_ctx_, fmt_ctx_->streams[audio_stream_idx_]->codecpar);
    avcodec_open2(audio_codec_ctx_, avcodec_find_decoder(fmt_ctx_->streams[audio_stream_idx_]->codecpar->codec_id), nullptr);

    size_t in_ch_layout = audio_codec_ctx_->channel_layout;
    if (!in_ch_layout) {
        in_ch_layout = av_get_default_channel_layout(audio_codec_ctx_->channels);
    }

    swr_ctx_ = swr_alloc_set_opts(
        nullptr,
        AV_CH_LAYOUT_MONO,  // output single channel
        AV_SAMPLE_FMT_FLT,  // output f32 dtype
        wav_sample_rate_,
        in_ch_layout,
        audio_codec_ctx_->sample_fmt,
        audio_codec_ctx_->sample_rate,
        0, nullptr);
    swr_init(swr_ctx_);
    if (!swr_is_initialized(swr_ctx_)) {
        printf("Resampler has not been properly initialized\n");
        return 1;
    }
    audio_frame_ = av_frame_alloc();

    return 0;
}
#else
int VideoDecoder::init_audio_decoder_macos() {
    // init audio decoder
    audio_codec_ctx_ = avcodec_alloc_context3(avcodec_find_decoder(fmt_ctx_->streams[audio_stream_idx_]->codecpar->codec_id));
    avcodec_parameters_to_context(audio_codec_ctx_, fmt_ctx_->streams[audio_stream_idx_]->codecpar);
    avcodec_open2(audio_codec_ctx_, avcodec_find_decoder(fmt_ctx_->streams[audio_stream_idx_]->codecpar->codec_id), nullptr);

    AVChannelLayout in_ch_layout;
    av_channel_layout_default(&in_ch_layout, audio_codec_ctx_->ch_layout.nb_channels);

    AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;
    SwrContext *swr = nullptr;
    int ret = swr_alloc_set_opts2(
        &swr,
        &out_ch_layout,  // output single channel
        AV_SAMPLE_FMT_FLT,  // output f32 dtype
        wav_sample_rate_,
        &in_ch_layout,
        audio_codec_ctx_->sample_fmt,
        audio_codec_ctx_->sample_rate,
        0, nullptr);
    if (ret < 0) {
        printf("Could not set resampler options\n");
        return 1;
    }

    // Initialize the resampling context
    ret = swr_init(swr);
    if (ret < 0) {
        printf("Could not initialize resampler\n");
        swr_free(&swr);
        return 1;
    }

    swr_ctx_ = swr;
    audio_frame_ = av_frame_alloc();

    return 0;
}
#endif
int VideoDecoder::init_video_decoder() {
    // init video decoder
    video_codec_ctx_ = avcodec_alloc_context3(avcodec_find_decoder(fmt_ctx_->streams[video_stream_idx_]->codecpar->codec_id));
    avcodec_parameters_to_context(video_codec_ctx_, fmt_ctx_->streams[video_stream_idx_]->codecpar);
    avcodec_open2(video_codec_ctx_, avcodec_find_decoder(fmt_ctx_->streams[video_stream_idx_]->codecpar->codec_id), nullptr);

    AVRational sar = fmt_ctx_->streams[video_stream_idx_]->sample_aspect_ratio;
    if (sar.num == 0)
        sar = av_make_q(1, 1);
    // cal dispaly ratio
    double dispaly_aspect_ration = (video_codec_ctx_->width * (double)sar.num) / (video_codec_ctx_->height * (double)sar.den);
    int target_width             = 0;
    int target_height            = 0;
    if (dispaly_aspect_ration >= 1.0) {
        target_width  = image_max_length_;
        target_height = std::lrint(target_width / dispaly_aspect_ration);
    } else {
        target_height = image_max_length_;
        target_width  = std::lrint(target_height * dispaly_aspect_ration);
    }

    // Make sure the width and height are even numbers (compatible with YUV420 and other formats)
    target_width += target_width % 2;
    target_height += target_height % 2;
    target_width  = std::max(target_width, 2);
    target_height = std::max(target_height, 2);

    sws_ctx_ = sws_getContext(
        video_codec_ctx_->width, video_codec_ctx_->height, video_codec_ctx_->pix_fmt,
        target_width, target_height, AV_PIX_FMT_RGB24,
        SWS_BICUBIC,
        nullptr, nullptr, nullptr);

    raw_frame_ = av_frame_alloc();
    rgb_frame_ = av_frame_alloc();

    rgb_frame_->format = AV_PIX_FMT_RGB24;
    rgb_frame_->width  = target_width;
    rgb_frame_->height = target_height;

    av_frame_get_buffer(rgb_frame_, 0);
    return 0;
}

VideoDecoder::VideoDecoder(std::string video_path) {
    if (avformat_open_input(&fmt_ctx_, video_path.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open file: " << video_path << "\n";
    }
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        std::cerr << "Could not find stream info\n";
    }

    for (size_t i = 0; i < fmt_ctx_->nb_streams; ++i) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
        } else if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx_ = i;
        }
    }

    if (audio_stream_idx_ == -1) {
        std::cerr << "No audio stream found\n";
    }
    if (video_stream_idx_ == -1) {
        std::cerr << "No video stream found\n";
    }

    this->init_video_decoder();
#ifdef MACOS_STREAM
    this->init_audio_decoder_macos();
#else
    this->init_audio_decoder();
#endif
}

void VideoDecoder::decode(double sec_interval) {
    AVPacket packet;
    double target_time = 1.0;
    while (av_read_frame(fmt_ctx_, &packet) >= 0) {
        if (packet.stream_index == video_stream_idx_) {
            avcodec_send_packet(video_codec_ctx_, &packet);
            while (avcodec_receive_frame(video_codec_ctx_, raw_frame_) == 0) {
                double pts_sec = raw_frame_->pts * av_q2d(fmt_ctx_->streams[video_stream_idx_]->time_base);
                if (pts_sec >= target_time) {
                    sws_scale(sws_ctx_, raw_frame_->data, raw_frame_->linesize, 0, raw_frame_->height,
                              rgb_frame_->data, rgb_frame_->linesize);
                    std::vector<uint8_t> buffer(rgb_frame_->width * rgb_frame_->height * 3);
                    for (size_t y = 0; y < rgb_frame_->height; ++y) {
                        std::memcpy(&buffer[y * rgb_frame_->width * 3],
                                    rgb_frame_->data[0] + y * rgb_frame_->linesize[0],
                                    rgb_frame_->width * 3);
                    }
                    image_buf<uint8_t> image;
                    image.nx  = rgb_frame_->width;
                    image.ny  = rgb_frame_->height;
                    image.buf = buffer;
                    video_buffers_.emplace_back(image);
                    target_time += sec_interval;
                }
            }
        } else if (packet.stream_index == audio_stream_idx_) {
            avcodec_send_packet(audio_codec_ctx_, &packet);
            while (avcodec_receive_frame(audio_codec_ctx_, audio_frame_) == 0) {
                int out_samples = swr_get_out_samples(swr_ctx_, audio_frame_->nb_samples);
                av_samples_alloc(converted_data_buf_, nullptr, out_nb_channels_, out_samples, AV_SAMPLE_FMT_FLT, 0);
                // resample
                out_samples = swr_convert(swr_ctx_, converted_data_buf_, out_samples, (const uint8_t**)audio_frame_->data, audio_frame_->nb_samples);
                if (out_samples > 0) {
                    int num_floats       = out_samples * out_nb_channels_;
                    const float* f32_ptr = reinterpret_cast<float*>(converted_data_buf_[0]);
                    pcmf32_data_.insert(pcmf32_data_.end(), f32_ptr, f32_ptr + num_floats);
                }
            }
        }
    }
    av_packet_unref(&packet);
    // TODO: neccesary?
    avcodec_send_packet(audio_codec_ctx_, nullptr);
    avcodec_send_packet(video_codec_ctx_, nullptr);
}

std::vector<image_buf<uint8_t>> VideoDecoder::get_video_buffer() {
    return video_buffers_;
}

std::vector<float> VideoDecoder::get_audio_pcmf32() {
    return pcmf32_data_;
}

static void write_little_endian(std::ofstream& file, uint32_t value) {
    char bytes[4];
    bytes[0] = static_cast<char>(value & 0xFF);
    bytes[1] = static_cast<char>((value >> 8) & 0xFF);
    bytes[2] = static_cast<char>((value >> 16) & 0xFF);
    bytes[3] = static_cast<char>((value >> 24) & 0xFF);
    file.write(bytes, 4);
}

static void write_little_endian_short(std::ofstream& file, uint16_t value) {
    char bytes[2];
    bytes[0] = static_cast<char>(value & 0xFF);
    bytes[1] = static_cast<char>((value >> 8) & 0xFF);
    file.write(bytes, 2);
}

// convert float to 4 bytes endian
static void write_float_sample(std::ofstream& file, float sample) {
    uint32_t asInt;
    std::memcpy(&asInt, &sample, sizeof(uint32_t));
    char bytes[4];
    bytes[0] = static_cast<char>(asInt & 0xFF);
    bytes[1] = static_cast<char>((asInt >> 8) & 0xFF);
    bytes[2] = static_cast<char>((asInt >> 16) & 0xFF);
    bytes[3] = static_cast<char>((asInt >> 24) & 0xFF);
    file.write(bytes, 4);
}

void VideoDecoder::write_wav(std::string file_name) {
    std::ofstream out_file(file_name, std::ios::binary);
    if (!out_file) {
        fprintf(stderr, "can not open %s\n", file_name.c_str());
        out_file.close();
        return;
    }

    const int bits_per_sample      = 16;
    const int byte_rate            = wav_sample_rate_ * out_nb_channels_ * bits_per_sample / 8;
    const int block_align          = out_nb_channels_ * bits_per_sample / 8;
    const uint32_t data_size       = pcmf32_data_.size() * sizeof(float);
    const uint32_t riff_chunk_size = 36 + data_size;

    // write RIFF header
    out_file.write("RIFF", 4);
    write_little_endian(out_file, riff_chunk_size);
    out_file.write("WAVE", 4);

    // write fmt subchunk size
    out_file.write("fmt ", 4);
    write_little_endian(out_file, 16);
    write_little_endian_short(out_file, 3);
    write_little_endian_short(out_file, out_nb_channels_);
    write_little_endian(out_file, wav_sample_rate_);
    write_little_endian(out_file, byte_rate);
    write_little_endian_short(out_file, block_align);
    write_little_endian_short(out_file, bits_per_sample);

    // write data header
    out_file.write("data", 4);
    write_little_endian(out_file, data_size);

    // write audio data
    for (const float sample : pcmf32_data_) {
        write_float_sample(out_file, sample);
    }

    out_file.close();
}

void VideoDecoder::write_jpg(std::string prefix_name = "") {
    if (prefix_name == "") {
        prefix_name = "image";
    }

    for (size_t i = 0; i < video_buffers_.size(); ++i) {
        auto image            = video_buffers_[i];
        std::string full_name = prefix_name + "_" + std::to_string(i) + ".jpg";
        stbi_write_jpg(full_name.c_str(), image.nx, image.ny, 3, image.buf.data(), 100);
    }
}
}  // namespace edge
