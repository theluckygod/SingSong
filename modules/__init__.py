import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from modules.audiolm_pytorch import AudioLM
from modules.soundstream import SoundStream, AudioLMSoundStream, MusicLMSoundStream
from modules.encodec import EncodecWrapper

from modules.audiolm_pytorch import SemanticTransformer, CoarseTransformer, FineTransformer
from modules.audiolm_pytorch import FineTransformerWrapper, CoarseTransformerWrapper, SemanticTransformerWrapper

from modules.vq_wav2vec import FairseqVQWav2Vec
from modules.hubert_kmeans import HubertWithKmeans

from modules.trainer import SoundStreamTrainer, SemanticTransformerTrainer, FineTransformerTrainer, CoarseTransformerTrainer

from modules.audiolm_pytorch import get_embeds
