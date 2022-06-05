from transformers.configuration_utils import PretrainedConfig

# Configuration class for image-text representation 
class ITRConfig(PretrainedConfig): 
    model_type = "ITR"

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        max_position_embeddings=40,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=384,
        patch_size=32,
        num_channels=3,
        qkv_bias=True,
        max_image_length=-1,
        tie_word_embeddings=False,
        num_images=-1,
        **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_image_length = max_image_length
        self.num_images = num_images



class IMITERConfig(PretrainedConfig): 
    model_type = "IMITER"

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        max_position_embeddings=40,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=384,
        patch_size=32,
        num_channels=3,
        qkv_bias=True,
        max_image_length=-1,
        tie_word_embeddings=False,
        num_images=-1, 
        imitate_bias=True, 
        **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_image_length = max_image_length
        self.num_images = num_images 

        self.imitate_bias = imitate_bias
