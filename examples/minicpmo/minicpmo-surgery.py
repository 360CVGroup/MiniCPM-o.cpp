# ruff: noqa
import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to MiniCPM-V model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
model = AutoModel.from_pretrained(
    args.model,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
)
checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith("resampler")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f"{args.model}/minicpmv.projector")

clip_tensors = [k for k, v in checkpoint.items() if k.startswith("vpm")]
if len(clip_tensors) > 0:
    clip = {name.replace("vpm.", ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/minicpmv.clip")

    # added tokens should be removed to be able to convert Mistral models
    if os.path.exists(f"{args.model}/added_tokens.json"):
        with open(f"{args.model}/added_tokens.json", "w") as f:
            f.write("{}\n")

config = model.llm.config
config.auto_map = {
    "AutoConfig": "configuration_minicpm.MiniCPMConfig",
    "AutoModel": "modeling_minicpm.MiniCPMModel",
    "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification",
}
model.llm.save_pretrained(f"{args.model}/model")
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.save_pretrained(f"{args.model}/model")

# get a list of mm tensor names
apm_mm_tensors = [
    k for k, v in checkpoint.items() if k.startswith("audio_projection_layer")
]

# store these tensors in a new dictionary and torch.save them
apm_projector = {name: checkpoint[name].float() for name in apm_mm_tensors}
# torch.save(projector, f"{args.model}/minicpmo-audio.projector")

whisper_tensors = [k for k, v in checkpoint.items() if k.startswith("apm")]
# append apm projector into a single file
whisper_tensors.extend(apm_projector)
if len(whisper_tensors) > 0:
    # whisper = {name.replace("apm.", ""): checkpoint[name].float() for name in whisper_tensors}
    whisper = {name: checkpoint[name].float() for name in whisper_tensors}
    torch.save(whisper, f"{args.model}/minicpmo.whisper")

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(
    f"Also, use {args.model}/minicpmv.projector to prepare a minicpmv-encoder.gguf file."
)
print(
    f"Also, use {args.model}/minicpmo.whisper to prepare a minicpmo-audio-encoder.gguf file."
)
