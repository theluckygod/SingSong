from modules import SoundStream, HubertWithKmeans, CoarseTransformer, FineTransformer, SingSong
import torch
import torchaudio


# define all dataset paths, checkpoints, etc
# downloaded at https://huggingface.co/haydenshively/SoundStream/tree/main
soundstream_ckpt = "checkpoints/soundstream/soundstream_variant_naturalspeech2.pt"

# downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
hubert_ckpt = "checkpoints/hubert/hubert_base_ls960.pt"
hubert_quantizer = "checkpoints/hubert/hubert_base_ls960_L9_km500.bin" # listed in row "HuBERT Base (~95M params)", column Quantizer


# Semantic codes
wav2vec = HubertWithKmeans(
    checkpoint_path = f"./{hubert_ckpt}",
    kmeans_path = f"./{hubert_quantizer}"
)

# Acoustic codes
soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

# Stage 1 + 2: Semantic-Acoustic Transformer
semantic_acoustic_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

# Stage 3: Fine Transformer
fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

# Everything together
singsong = SingSong(
    wav2vec = wav2vec,
    codec = soundstream,
    sa_transformer = semantic_acoustic_transformer,
    fine_transformer = fine_transformer
)


generated_wav = singsong(prime_wave=torch.randn(1, 320 * 8), batch_size=1)

output_path = "out.wav"
sample_rate = 44100
torchaudio.save(output_path, generated_wav.cpu(), sample_rate)