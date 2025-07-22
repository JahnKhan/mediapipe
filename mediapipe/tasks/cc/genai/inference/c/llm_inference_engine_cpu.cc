// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pthread.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/proto/transformer_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_builder_factory.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
// clang-format off
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"
// clang-format on
#include "sentencepiece/src/sentencepiece_processor.h"  // from @com_google_sentencepiece
#include "sentencepiece/src/util.h"  // from @com_google_sentencepiece
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "mediapipe/util/audio_decoder.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/calculators/tensor/audio_to_tensor_calculator.h"

namespace {

using ::mediapipe::tasks::genai::llm_utils::ScopedFile;

// Audio processing constants based on Gemini specifications
constexpr int kTargetSampleRate = 16000;  // 16kHz as required by Gemini
constexpr int kTargetChannels = 1;        // Mono audio
constexpr int kTokensPerSecond = 32;      // Gemini: 32 tokens per second of audio
constexpr int kMaxAudioLengthSeconds = 9 * 3600 + 30 * 60;  // 9.5 hours max

// Audio format validation
bool IsValidAudioFormat(const std::string& audio_data) {
  if (audio_data.size() < 8) {
    return false;
  }
  
  // Check for common audio file headers
  // WAV: "RIFF" and "WAVE"
  if (audio_data.substr(0, 4) == "RIFF" && 
      audio_data.size() > 12 && 
      audio_data.substr(8, 4) == "WAVE") {
    return true;
  }
  
  // MP3: check for MP3 frame sync
  if (audio_data.size() >= 2) {
    unsigned char byte1 = static_cast<unsigned char>(audio_data[0]);
    unsigned char byte2 = static_cast<unsigned char>(audio_data[1]);
    if (byte1 == 0xFF && (byte2 & 0xE0) == 0xE0) {
      return true;
    }
  }
  
  // Basic checks for other formats could be added here
  return false;
}

// Audio format validation and preprocessing
struct AudioPreprocessingResult {
  std::vector<float> audio_samples;
  int sample_rate;
  int channels;
  int duration_ms;
  int token_count;
};

absl::StatusOr<AudioPreprocessingResult> PreprocessAudioData(
    const std::string& audio_data) {
  AudioPreprocessingResult result;
  
  // Validate audio format
  if (!IsValidAudioFormat(audio_data)) {
    return absl::InvalidArgumentError(
        "Unsupported audio format. Supported formats: WAV, MP3, AAC, OGG, FLAC");
  }
  
  // Decode audio data (supports WAV, MP3, AAC, etc.)
  auto decoder_result = mediapipe::DecodeAudioFromMemory(
      audio_data.data(), audio_data.size());
  
  if (!decoder_result.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to decode audio data: ", decoder_result.status().message()));
  }
  
  auto& [audio_matrix, sample_rate] = decoder_result.value();
  
  // Validate audio parameters
  if (sample_rate <= 0 || sample_rate > 192000) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid sample rate: ", sample_rate, "Hz. Expected range: 1-192000Hz"));
  }
  
  int input_channels = audio_matrix.rows();
  int num_samples = audio_matrix.cols();
  
  if (input_channels <= 0 || input_channels > 8) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid number of channels: ", input_channels, 
                    ". Expected range: 1-8 channels"));
  }
  
  if (num_samples <= 0) {
    return absl::InvalidArgumentError("Audio contains no samples");
  }
  
  // Calculate duration and validate length
  float duration_seconds = static_cast<float>(num_samples) / sample_rate;
  if (duration_seconds > kMaxAudioLengthSeconds) {
    return absl::InvalidArgumentError(
        absl::StrCat("Audio too long: ", duration_seconds, 
                    " seconds (max: ", kMaxAudioLengthSeconds, " seconds)"));
  }
  
  if (duration_seconds < 0.1f) {
    return absl::InvalidArgumentError(
        absl::StrCat("Audio too short: ", duration_seconds, 
                    " seconds (min: 0.1 seconds)"));
  }

  // Convert to mono if necessary
  mediapipe::Matrix mono_audio;
  if (input_channels == 1) {
    mono_audio = audio_matrix;
  } else {
    // Convert multi-channel to mono by averaging
    mono_audio = mediapipe::Matrix::Zero(1, num_samples);
    for (int i = 0; i < num_samples; ++i) {
      float sum = 0.0f;
      for (int ch = 0; ch < input_channels; ++ch) {
        sum += audio_matrix(ch, i);
      }
      mono_audio(0, i) = sum / input_channels;
    }
    ABSL_LOG(INFO) << "Converted " << input_channels << " channels to mono";
  }
  
  // Resample to 16kHz if necessary
  mediapipe::Matrix resampled_audio;
  int output_sample_rate = sample_rate;
  
  if (sample_rate != kTargetSampleRate) {
    ABSL_LOG(INFO) << "Resampling from " << sample_rate << "Hz to " << kTargetSampleRate << "Hz";
    
    // Calculate new sample count
    int new_sample_count = static_cast<int>(
        num_samples * static_cast<float>(kTargetSampleRate) / sample_rate);
    
    resampled_audio = mediapipe::Matrix::Zero(1, new_sample_count);
    
    // Simple linear interpolation resampling
    float ratio = static_cast<float>(num_samples) / new_sample_count;
    for (int i = 0; i < new_sample_count; ++i) {
      float src_index = i * ratio;
      int src_int = static_cast<int>(src_index);
      float frac = src_index - src_int;
      
      if (src_int + 1 < num_samples) {
        resampled_audio(0, i) = mono_audio(0, src_int) * (1.0f - frac) + 
                               mono_audio(0, src_int + 1) * frac;
      } else {
        resampled_audio(0, i) = mono_audio(0, src_int);
      }
    }
    output_sample_rate = kTargetSampleRate;
    num_samples = new_sample_count;
  } else {
    resampled_audio = mono_audio;
  }
  
  // Convert to vector format
  result.audio_samples.resize(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    result.audio_samples[i] = resampled_audio(0, i);
  }
  
  result.sample_rate = output_sample_rate;
  result.channels = 1;
  result.duration_ms = static_cast<int>(duration_seconds * 1000);
  result.token_count = static_cast<int>(duration_seconds * kTokensPerSecond);
  
  ABSL_LOG(INFO) << "Audio preprocessed: " << duration_seconds << "s, " 
                 << result.token_count << " tokens, " << output_sample_rate << "Hz";
  
  return result;
}

// Convert audio samples to tokens for LLM processing
std::vector<int> AudioSamplesToTokens(const std::vector<float>& audio_samples,
                                     int sample_rate) {
  // This is a simplified audio tokenization
  // In a complete implementation, this would use a proper audio encoder
  // like Whisper encoder or similar multimodal tokenization
  
  std::vector<int> audio_tokens;
  
  // Group audio into frames (32ms frames at 16kHz = 512 samples per frame)
  int samples_per_frame = (sample_rate * 32) / 1000;  // 32ms frames
  int num_frames = audio_samples.size() / samples_per_frame;
  
  // Reserve space for tokens (approximately 1 token per frame)
  audio_tokens.reserve(num_frames);
  
  // Generate representative tokens for each frame
  for (int frame = 0; frame < num_frames; ++frame) {
    int start_idx = frame * samples_per_frame;
    int end_idx = std::min(start_idx + samples_per_frame, 
                          static_cast<int>(audio_samples.size()));
    
    // Calculate RMS energy for this frame
    float rms = 0.0f;
    for (int i = start_idx; i < end_idx; ++i) {
      rms += audio_samples[i] * audio_samples[i];
    }
    rms = std::sqrt(rms / (end_idx - start_idx));
    
    // Map RMS energy to a token ID range (simplified approach)
    // In practice, this would be done by a trained audio encoder
    int token_id = 50000 + static_cast<int>(rms * 1000) % 10000;
    audio_tokens.push_back(token_id);
  }
  
  return audio_tokens;
}

constexpr int kCheckLastKChars = 10;

struct TfLiteLlm {
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<mediapipe::tasks::core::ModelAssetBundleResources> resources;
};

struct LlmInferenceEngineCpu_Engine {
  const sentencepiece::SentencePieceProcessor* tokenizer;
  const absl::flat_hash_map<unsigned char, int>* bytes_to_unicode_mapper;
  const absl::flat_hash_map<int, unsigned char>* unicode_to_bytes_mapper;
  const std::variant<mediapipe::tasks::genai::xnn_utils::Llm*, TfLiteLlm*> llm;
  const int start_token_id;
  const std::vector<std::string> stop_tokens;
  const size_t max_num_tokens;

  ~LlmInferenceEngineCpu_Engine() {
    delete tokenizer;
    delete bytes_to_unicode_mapper;
    delete unicode_to_bytes_mapper;
    if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(llm)) {
      delete std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(llm);
    } else {
      delete std::get<TfLiteLlm*>(llm);
    }
  };
};

struct LlmInferenceEngineCpu_Session {
  const LlmInferenceEngineCpu_Engine* engine;
  std::string prompt;
  std::string audio_data;
  std::vector<int> audio_tokens;  // Preprocessed audio tokens
  bool has_audio_input;
  int timestep;
  std::string last_10_char;
  std::string final_output;
  std::function<void(std::string)> cpu_callback;
  bool early_stop;
  pthread_t work_id;
  int next_token_id;
  ~LlmInferenceEngineCpu_Session() { pthread_join(work_id, nullptr); };
};

absl::StatusOr<std::unique_ptr<absl::flat_hash_map<unsigned char, int>>>
CreateBytesToUnicodeMapper() {
  auto bytes_to_unicode_mapper =
      std::make_unique<absl::flat_hash_map<unsigned char, int>>();
  // "!" - "~"
  for (int i = 33; i <= 126; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  // "¡" - "¬"
  for (int i = 161; i <= 172; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  // "®" - "ÿ"
  for (int i = 174; i < 256; i++) {
    bytes_to_unicode_mapper->insert({static_cast<uint8_t>(i), i});
  }
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (!bytes_to_unicode_mapper->contains(static_cast<uint8_t>(b))) {
      bytes_to_unicode_mapper->insert({static_cast<uint8_t>(b), 256 + n});
      n += 1;
    }
  }
  return bytes_to_unicode_mapper;
}

absl::StatusOr<std::unique_ptr<absl::flat_hash_map<int, unsigned char>>>
CreateUnicodeToBytesMapper() {
  MP_ASSIGN_OR_RETURN(auto bytes_to_unicode_mapper,
                      CreateBytesToUnicodeMapper());
  auto unicode_to_bytes_mapper =
      std::make_unique<absl::flat_hash_map<int, unsigned char>>();
  for (const auto& [key, value] : *bytes_to_unicode_mapper) {
    unicode_to_bytes_mapper->insert({value, key});
  }
  return unicode_to_bytes_mapper;
}

std::string MapBytesToUnicode(
    absl::string_view prompt,
    const absl::flat_hash_map<unsigned char, int>* bytes_to_unicode_mapper) {
  std::string converted_prompt = "";
  for (const uint8_t byte : prompt) {
    converted_prompt.append(sentencepiece::string_util::UnicodeCharToUTF8(
        bytes_to_unicode_mapper->at(byte)));
  }
  return converted_prompt;
}

std::string MapUnicodeToBytes(
    absl::string_view output,
    const absl::flat_hash_map<int, uint8_t>* unicode_to_bytes_mapper) {
  sentencepiece::string_util::UnicodeText unicode_text =
      sentencepiece::string_util::UTF8ToUnicodeText(output);
  std::string converted_output = "";
  for (const int code_point : unicode_text) {
    if (!unicode_to_bytes_mapper->contains(code_point)) {
      converted_output += code_point;
    } else {
      converted_output += unicode_to_bytes_mapper->at(code_point);
    }
  }
  return converted_output;
}

void* next_token_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;
  if (cpu_session->timestep < cpu_session->engine->max_num_tokens) {
    if (cpu_session->early_stop) {
      return nullptr;
    }

    auto token_ids_per_step = std::vector<int>();
    if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(
            cpu_session->engine->llm)) {
      auto status = std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(
                        cpu_session->engine->llm)
                        ->GetNextToken(&token_ids_per_step);
      if (!status.ok()) {
        ABSL_LOG(FATAL) << "Failed to generate output: " << status;
      }
    } else {
      auto llm = std::get<TfLiteLlm*>(cpu_session->engine->llm);
      auto* decode_runner = llm->interpreter->GetSignatureRunner("decode");
      ABSL_CHECK_EQ(decode_runner->AllocateTensors(), kTfLiteOk);
      TfLiteTensor* decode_input = decode_runner->input_tensor("args_0");
      TfLiteTensor* decode_input_pos = decode_runner->input_tensor("args_1");
      decode_input->data.i64[0] =
          static_cast<int64_t>(cpu_session->next_token_id);
      decode_input_pos->data.i64[0] =
          static_cast<int64_t>(cpu_session->timestep);

      // logits->dims->data[0] = batch size
      // logits->dims->data[1] = sequence length
      // logits->dims->data[2] = vocab size
      const TfLiteTensor* logits = decode_runner->output_tensor("output_0");

      ABSL_CHECK_EQ(decode_runner->Invoke(), kTfLiteOk);

      auto max_logit_it = std::max_element(
          logits->data.f, logits->data.f + logits->dims->data[2]);
      token_ids_per_step.push_back(std::distance(logits->data.f, max_logit_it));
    }

    // For future multithreading support.
    if (cpu_session->early_stop) {
      return nullptr;
    }

    if (cpu_session->timestep >= cpu_session->engine->max_num_tokens) {
      cpu_session->early_stop = true;
    }

    cpu_session->next_token_id = token_ids_per_step[0];

    std::string token =
        cpu_session->engine->tokenizer->IdToPiece(token_ids_per_step[0]);
    if (cpu_session->engine->unicode_to_bytes_mapper != nullptr) {
      token = MapUnicodeToBytes(token,
                                cpu_session->engine->unicode_to_bytes_mapper);
    } else {
      token = absl::StrReplaceAll(token, {{"▁", " "}});
    }
    cpu_session->last_10_char.append(token);

    int stop_index;
    for (const auto& stop_token : cpu_session->engine->stop_tokens) {
      stop_index = cpu_session->last_10_char.find(stop_token);
      if (stop_index != std::string::npos) {
        cpu_session->early_stop = true;
        cpu_session->last_10_char =
            cpu_session->last_10_char.substr(0, stop_index);
        break;
      }
    }

    std::string ready_char = "";
    if (cpu_session->early_stop) {
      ready_char = cpu_session->last_10_char;
    } else if (cpu_session->last_10_char.size() > kCheckLastKChars) {
      ready_char = cpu_session->last_10_char.substr(
          0, cpu_session->last_10_char.size() - kCheckLastKChars);
      cpu_session->last_10_char = cpu_session->last_10_char.substr(
          cpu_session->last_10_char.size() - kCheckLastKChars);
    }
    cpu_session->final_output.append(ready_char);

    cpu_session->cpu_callback(ready_char);

    ++cpu_session->timestep;

    next_token_function(args);
  }
  return nullptr;
};

void* start_llm_function(void* args) {
  struct LlmInferenceEngineCpu_Session* cpu_session =
      (struct LlmInferenceEngineCpu_Session*)args;

  std::vector<int> prompt_ids = {};

  std::string prompt;

  // Process audio input if available
  if (cpu_session->has_audio_input && !cpu_session->audio_tokens.empty()) {
    ABSL_LOG(INFO) << "Processing audio input with " 
                   << cpu_session->audio_tokens.size() << " audio tokens";
    
    // Add audio tokens to the prompt_ids
    // In a complete implementation, audio tokens would be properly integrated
    // with text tokens according to the model's multimodal architecture
    prompt_ids.insert(prompt_ids.end(), 
                     cpu_session->audio_tokens.begin(), 
                     cpu_session->audio_tokens.end());
  }

  if (cpu_session->engine->bytes_to_unicode_mapper != nullptr) {
    prompt = MapBytesToUnicode(cpu_session->prompt,
                               cpu_session->engine->bytes_to_unicode_mapper);
  } else {
    prompt = cpu_session->prompt;
  }

  // Tokenize text prompt
  std::vector<int> text_prompt_ids;
  auto status = cpu_session->engine->tokenizer->Encode(prompt, &text_prompt_ids);

  if (!status.ok()) {
    ABSL_LOG(FATAL) << "Failed to encode input: " << status;
  }
  
  // Combine audio and text tokens
  // For multimodal models, the order and integration of tokens matters
  // This is a simplified approach - in practice, this would follow
  // the specific model's multimodal token organization
  if (cpu_session->has_audio_input) {
    // Insert text tokens after audio tokens (model-dependent)
    prompt_ids.insert(prompt_ids.end(), 
                     text_prompt_ids.begin(), 
                     text_prompt_ids.end());
    
    ABSL_LOG(INFO) << "Combined tokens: " 
                   << cpu_session->audio_tokens.size() << " audio + "
                   << text_prompt_ids.size() << " text = "
                   << prompt_ids.size() << " total";
  } else {
    prompt_ids = text_prompt_ids;
  }
  
  prompt_ids.insert(prompt_ids.begin(), cpu_session->engine->start_token_id);

  if (std::holds_alternative<mediapipe::tasks::genai::xnn_utils::Llm*>(
          cpu_session->engine->llm)) {
    auto llm = std::get<mediapipe::tasks::genai::xnn_utils::Llm*>(
        cpu_session->engine->llm);
    ABSL_CHECK_OK(llm->SeekTimeStep(0));
    ABSL_CHECK_OK(llm->AddInputTokens({prompt_ids}));
  } else {
    auto llm = std::get<TfLiteLlm*>(cpu_session->engine->llm);
    auto* prefill_runner = llm->interpreter->GetSignatureRunner("prefill");

    ABSL_CHECK_EQ(prefill_runner->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* prefill_input = prefill_runner->input_tensor("args_0");
    TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("args_1");
    memset(prefill_input->data.data, 0, prefill_input->bytes);
    memset(prefill_input_pos->data.data, 0, prefill_input_pos->bytes);
    cpu_session->next_token_id = prompt_ids.back();
    prompt_ids.pop_back();
    for (int i = 0; i < prompt_ids.size(); ++i) {
      prefill_input->data.i64[i] = static_cast<int64_t>(prompt_ids[i]);
      prefill_input_pos->data.i64[i] = static_cast<int64_t>(i);
    }
    ABSL_CHECK_EQ(prefill_runner->Invoke(), kTfLiteOk);
  }

  cpu_session->timestep = prompt_ids.size();

  next_token_function(args);

  return nullptr;
}

absl::StatusOr<std::unique_ptr<LlmInferenceEngineCpu_Engine>>
CreateXnnLlmCpuEngine(const LlmModelSettings* model_settings) {
  MP_ASSIGN_OR_RETURN(auto model_file,
                      ScopedFile::Open(model_settings->model_path));
  MP_ASSIGN_OR_RETURN(auto model_data,
                      mediapipe::tasks::genai::llm_utils::ModelData::Create(
                          std::move(model_file)));

  if (model_settings->number_of_supported_lora_ranks != 0) {
    ABSL_LOG(FATAL) << "LoRA on CPU is not supported yet.";
  }

  auto llm_params_proto = model_data->GetLlmParameters();
  auto llm_params =
      mediapipe::tasks::genai::xnn_utils::LlmParams::FromLLMParametersProto(
          llm_params_proto);

  auto model_type = model_data->GetModelType();
  RET_CHECK(model_type) << "Failed to get model type.";

  MP_ASSIGN_OR_RETURN(auto backend,
                      model_data->ReadMetadata(
                          mediapipe::tasks::genai::llm_utils::kLlmBackendName));
  RET_CHECK_EQ(backend, "cpu");

  // Create directory for tokenizer and model cache file.
  if (model_settings->cache_dir != nullptr) {
    auto s = mediapipe::file::RecursivelyCreateDir(model_settings->cache_dir);
    if (!s.ok()) {
      ABSL_LOG(WARNING) << s;
    }
  }

  MP_ASSIGN_OR_RETURN(auto spm_model_content,
                      model_data->ReadMetadata("spm_vocab_model"));
  model_data.reset();

  llm_params.seq_size_T = model_settings->max_num_tokens;
  llm_params.cache_dir = model_settings->cache_dir;

  auto weight_loader = std::make_unique<
      mediapipe::tasks::genai::xnn_utils::DefaultLlmWeightsLoader>(
      model_settings->model_path, llm_params);

  auto runtime_configs =
      std::make_unique<mediapipe::tasks::genai::xnn_utils::RuntimeConfigs>();

  MP_ASSIGN_OR_RETURN(auto llm,
                      mediapipe::tasks::genai::xnn_utils::CreateLlm(
                          llm_params, std::move(runtime_configs),
                          std::move(weight_loader), nullptr, *model_type));

  auto tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MP_RETURN_IF_ERROR(tokenizer->LoadFromSerializedProto(spm_model_content));

  std::unique_ptr<absl::flat_hash_map<unsigned char, int>>
      bytes_to_unicode_mapper;
  std::unique_ptr<absl::flat_hash_map<int, unsigned char>>
      unicode_to_bytes_mapper;
  // These models uses GPT2 style unicode mapping, which additional mapping is
  // needed.
  if (model_type == odml::infra::proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B ||
      model_type == odml::infra::proto::LLM_MODEL_TYPE_FALCON_RW_1B ||
      model_type == odml::infra::proto::LLM_MODEL_TYPE_PHI_2) {
    MP_ASSIGN_OR_RETURN(bytes_to_unicode_mapper, CreateBytesToUnicodeMapper());
    MP_ASSIGN_OR_RETURN(unicode_to_bytes_mapper, CreateUnicodeToBytesMapper());
  }

  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine(
      new LlmInferenceEngineCpu_Engine{
          .tokenizer = tokenizer.release(),
          .bytes_to_unicode_mapper = bytes_to_unicode_mapper.release(),
          .unicode_to_bytes_mapper = unicode_to_bytes_mapper.release(),
          .llm = llm.release(),
          .start_token_id = llm_params_proto.start_token_id(),
          .stop_tokens =
              std::vector<std::string>(llm_params_proto.stop_tokens().begin(),
                                       llm_params_proto.stop_tokens().end()),
          .max_num_tokens = model_settings->max_num_tokens,
      });

  return engine;
}

// Creates an inference engine from a *.task file.
// This method extracts the TF_LITE_PREFILL_DECODE, TOKENIZER_MODEL and METADATA
// files from the task bundle and initializes the TfLIte XNNPack delegate.
absl::StatusOr<std::unique_ptr<LlmInferenceEngineCpu_Engine>>
CreateTfliteLlmCpuEngine(const LlmModelSettings* model_settings) {
  auto external_file =
      std::make_unique<mediapipe::tasks::core::proto::ExternalFile>();
  if (model_settings) {
    external_file->set_file_name(model_settings->model_path);
  }
  MP_ASSIGN_OR_RETURN(auto resources,
                      mediapipe::tasks::core::ModelAssetBundleResources::Create(
                          "", std::move(external_file)));
  const std::vector<std::string>& files_list = resources->ListFiles();
  const absl::flat_hash_set<std::string> files_set(files_list.begin(),
                                                   files_list.end());

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (!files_set.contains("TF_LITE_PREFILL_DECODE")) {
    return absl::InvalidArgumentError("TF_LITE_PREFILL_DECODE not found.");
  }
  if (!files_set.contains("TOKENIZER_MODEL")) {
    return absl::InvalidArgumentError("TOKENIZER_MODEL not found.");
  }
  if (!files_set.contains("METADATA")) {
    return absl::InvalidArgumentError("METADATA not found.");
  }
  MP_ASSIGN_OR_RETURN(absl::string_view model_buffer,
                      resources->GetFile("TF_LITE_PREFILL_DECODE"));
  MP_ASSIGN_OR_RETURN(absl::string_view tokenizer_buffer,
                      resources->GetFile("TOKENIZER_MODEL"));
  MP_ASSIGN_OR_RETURN(absl::string_view params_buffer,
                      resources->GetFile("METADATA"));
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_buffer.data(),
                                                        model_buffer.size());
  RET_CHECK(model) << "Failed to build TF_LITE_PREFILL_DECODE model.";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  RET_CHECK(model_settings);
  builder(&interpreter);
  RET_CHECK_NE(interpreter, nullptr);

  // RET_CHECK(model_settings->xnnpack_options.has_value());
  auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
  // Set the number of threads to 4 as default.
  delegate_options.num_threads = 4;
  // Compute the path for the cache file.
  std::string weight_cache_path = model_settings->cache_dir;
  if (weight_cache_path != ":nocache") {
    if (weight_cache_path.empty()) {
      weight_cache_path =
          absl::StrCat(model_settings->model_path, ".xnnpack_cache");
    } else {
      weight_cache_path = mediapipe::file::JoinPath(
          weight_cache_path,
          absl::StrCat(mediapipe::file::Basename(model_settings->model_path),
                       ".xnnpack_cache"));
    }
    delegate_options.weight_cache_file_path = weight_cache_path.c_str();
  }
  RET_CHECK_EQ(interpreter->ModifyGraphWithDelegate(
                   tflite::Interpreter::TfLiteDelegatePtr(
                       TfLiteXNNPackDelegateCreate(&delegate_options),
                       [](TfLiteDelegate* delegate) {
                         TfLiteXNNPackDelegateDelete(delegate);
                       })),
               kTfLiteOk);
  RET_CHECK_EQ(interpreter->SetNumThreads(4), kTfLiteOk);

  auto tflite_llm = std::make_unique<TfLiteLlm>(
      TfLiteLlm{std::move(interpreter), std::move(resources)});

  auto tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MP_RETURN_IF_ERROR(tokenizer->LoadFromSerializedProto(tokenizer_buffer));

  auto llm_parameters = odml::infra::proto::LlmParameters();
  RET_CHECK(llm_parameters.ParseFromArray(params_buffer.data(),
                                          params_buffer.size()));

  auto start_token_id = tokenizer->PieceToId(llm_parameters.start_token());

  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine(
      new LlmInferenceEngineCpu_Engine{
          .tokenizer = tokenizer.release(),
          .bytes_to_unicode_mapper = nullptr,
          .unicode_to_bytes_mapper = nullptr,
          .llm = tflite_llm.release(),
          .start_token_id = start_token_id,
          .stop_tokens =
              std::vector<std::string>(llm_parameters.stop_tokens().begin(),
                                       llm_parameters.stop_tokens().end()),
          .max_num_tokens = model_settings->max_num_tokens,
      });

  return engine;
}

absl::StatusOr<LlmInferenceEngine_Engine*>
LlmInferenceEngine_CreateEngine_Helper(const LlmModelSettings* model_settings) {
  std::unique_ptr<LlmInferenceEngineCpu_Engine> engine;
  if (absl::EndsWith(model_settings->model_path, ".tflite")) {
    MP_ASSIGN_OR_RETURN(engine, CreateXnnLlmCpuEngine(model_settings));
  } else {
    MP_ASSIGN_OR_RETURN(engine, CreateTfliteLlmCpuEngine(model_settings));
  }

  return engine.release();
}

absl::StatusOr<LlmInferenceEngine_Session*>
LlmInferenceEngine_CreateSession_Helper(
    const LlmInferenceEngineCpu_Engine* engine,
    const LlmSessionConfig* session_config) {
  std::unique_ptr<LlmInferenceEngineCpu_Session> session(
      new LlmInferenceEngineCpu_Session{
          .engine = engine, 
          .audio_data = "",
          .audio_tokens = {},
          .has_audio_input = false
      });

  // Configure audio modality if enabled
  if (session_config && session_config->enable_audio_modality) {
    ABSL_LOG(INFO) << "Audio modality enabled for session";
    
    // Validate audio configuration
    if (engine->enable_audio_modality) {
      ABSL_LOG(INFO) << "Engine supports audio modality, max sequence length: " 
                     << engine->max_audio_sequence_length;
    } else {
      ABSL_LOG(WARNING) << "Session requests audio modality but engine doesn't support it";
    }
  }

  return session.release();
}

}  // namespace

void LlmInferenceEngine_CloseResponseContext(
    LlmResponseContext* response_context) {
  for (size_t i = 0; i < response_context->response_count; i++) {
    free(const_cast<char*>(response_context->response_array[i]));
  }
  free(response_context->response_array);
  response_context->response_array = nullptr;
  response_context->response_count = 0;
}

int LlmInferenceEngine_CreateEngine(const LlmModelSettings* model_settings,
                                    LlmInferenceEngine_Session** engine_out,
                                    char** error_msg) {
  auto engine = LlmInferenceEngine_CreateEngine_Helper(model_settings);
  if (!engine.ok()) {
    if (error_msg) {
      *error_msg = strdup(
          absl::StrCat("Failed to create engine: ", engine.status().ToString())
              .c_str());
    }
    return static_cast<int>(engine.status().code());
  }
  *engine_out = engine.value();
  return 0;
}

void LlmInferenceEngine_Engine_Delete(LlmInferenceEngine_Engine* engine) {
  delete reinterpret_cast<LlmInferenceEngineCpu_Engine*>(engine);
}

int LlmInferenceEngine_CreateSession(LlmInferenceEngine_Engine* engine,
                                     const LlmSessionConfig* session_config,
                                     LlmInferenceEngine_Session** session_out,
                                     char** error_msg) {
  auto cpu_engine = reinterpret_cast<LlmInferenceEngineCpu_Engine*>(engine);
  auto session =
      LlmInferenceEngine_CreateSession_Helper(cpu_engine, session_config);
  if (!session.ok()) {
    if (error_msg) {
      *error_msg = strdup(absl::StrCat("Failed to create session: ",
                                       session.status().ToString())
                              .c_str());
    }
    return static_cast<int>(session.status().code());
  }
  *session_out = session.value();
  return 0;
}

int LlmInferenceEngine_Session_Delete(LlmInferenceEngine_Session* session) {
  delete reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  return 0;
}

int LlmInferenceEngine_Session_AddQueryChunk(
    LlmInferenceEngine_Session* session, const char* input, char** error_msg) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  cpu_session->prompt = input;
  return 0;
}

ODML_EXPORT int LlmInferenceEngine_Session_AddImage(
    LlmInferenceEngine_Session* session, const void* sk_bitmap,
    char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

ODML_EXPORT int LlmInferenceEngine_Session_AddAudio(
    LlmInferenceEngine_Engine* engine, LlmInferenceEngine_Session* session,
    const char* audio_bytes, int audio_bytes_size, char** error_msg) {
  
  if (!audio_bytes || audio_bytes_size <= 0) {
    *error_msg = strdup("Invalid audio data provided");
    return 1;
  }
  
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  
  // Store raw audio data
  cpu_session->audio_data = std::string(audio_bytes, audio_bytes_size);
  
  // Preprocess audio data
  auto preprocessing_result = PreprocessAudioData(cpu_session->audio_data);
  
  if (!preprocessing_result.ok()) {
    std::string error = absl::StrCat("Audio preprocessing failed: ", 
                                   preprocessing_result.status().message());
    *error_msg = strdup(error.c_str());
    ABSL_LOG(ERROR) << error;
    return 1;
  }
  
  auto& audio_result = preprocessing_result.value();
  
  // Convert audio to tokens
  cpu_session->audio_tokens = AudioSamplesToTokens(
      audio_result.audio_samples, audio_result.sample_rate);
  
  cpu_session->has_audio_input = true;
  
  ABSL_LOG(INFO) << "Audio added successfully: " 
                 << audio_result.duration_ms << "ms, "
                 << cpu_session->audio_tokens.size() << " tokens";
  
  return 0;
}

int LlmInferenceEngine_Session_PredictSync(LlmInferenceEngine_Session* session,
                                           LlmResponseContext* response_context,
                                           char** error_msg) {
  auto status = LlmInferenceEngine_Session_PredictAsync(
      session, nullptr, error_msg,
      [](void* callback_context, LlmResponseContext* response_context) {});
  if (status != 0) {
    return status;
  }

  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  pthread_join(cpu_session->work_id, nullptr);
  cpu_session->work_id = 0;
  auto final_output = cpu_session->final_output;

  char** result = (char**)malloc(sizeof(char*) * 1);
  if (result == nullptr) {
    *error_msg = strdup("Failed to allocate result for cpu session.");
    return static_cast<int>(absl::StatusCode::kResourceExhausted);
  }

  result[0] = (char*)malloc(sizeof(char*) * (final_output.size() + 1));
  if (result[0] == nullptr) {
    *error_msg = strdup("Failed to allocate result for cpu session.");
    return static_cast<int>(absl::StatusCode::kResourceExhausted);
  }

  snprintf(result[0], final_output.size() + 1, "%s", final_output.c_str());

  response_context->response_array = result;
  response_context->response_count = 1;
  response_context->done = true;

  return 0;
}

int LlmInferenceEngine_Session_PredictAsync(
    LlmInferenceEngine_Session* session, void* callback_context,
    char** error_msg,
    void (*callback)(void* callback_context,
                     LlmResponseContext* response_context)) {
  if (session == nullptr) {
    *error_msg = strdup("Session is null.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }
  if (callback == nullptr) {
    *error_msg = strdup("Callback is null.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }

  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);

  if (cpu_session == nullptr) {
    *error_msg = strdup("Provided session is not a CPU session.");
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }

  cpu_session->cpu_callback = [=](std::string responses) -> void {
    char** result = (char**)malloc(sizeof(char*) * 1);
    if (result == nullptr) {
      ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
    }

    result[0] = (char*)malloc(sizeof(char*) * (responses.size() + 1));
    if (result[0] == nullptr) {
      ABSL_LOG(FATAL) << "Failed to allocate result for cpu session.";
    }

    snprintf(result[0], responses.size() + 1, "%s", responses.c_str());
    auto response_context = std::make_unique<LlmResponseContext>();
    response_context->response_array = result,
    response_context->response_count = 1,
    response_context->done = cpu_session->early_stop;
    callback(callback_context, response_context.release());
  };

  cpu_session->final_output = "";
  cpu_session->last_10_char = "";
  cpu_session->early_stop = false;

  pthread_t work_id = 0;
  cpu_session->work_id = work_id;
  pthread_create(&cpu_session->work_id, nullptr, start_llm_function,
                 cpu_session);

  return 0;
}

int LlmInferenceEngine_Session_PendingProcessCancellation(
    LlmInferenceEngine_Session* session, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_Session_Clone(
    LlmInferenceEngine_Session* session,
    LlmInferenceEngine_Session** cloned_session, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_Session_SizeInTokens(LlmInferenceEngine_Session* session,
                                            const char* input,
                                            char** error_msg) {
  auto cpu_session = reinterpret_cast<LlmInferenceEngineCpu_Session*>(session);
  std::vector<int> output_ids;
  auto status = cpu_session->engine->tokenizer->Encode(input, &output_ids);
  if (!status.ok()) {
    *error_msg = strdup(status.ToString().c_str());
    return -1;
  }
  return output_ids.size();
}

int LlmInferenceEngine_UpdateRuntimeConfig(LlmInferenceEngine_Session* session,
                                           const SessionRuntimeConfig* config,
                                           char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}

int LlmInferenceEngine_GetSentencePieceProcessor(
    LlmInferenceEngine_Engine* engine,
    const SentencePieceProcessor** processor_out, char** error_msg) {
  *error_msg = strdup("Not implemented");
  return 12;
}
