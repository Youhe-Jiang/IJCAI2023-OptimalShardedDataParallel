from osdp_utils.operator_splitting import Linear
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

def opt_osdp_model(model, hidden_size, slice_size):
    model.model.decoder.embed_tokens = FSDP(model.model.decoder.embed_tokens)
    model.model.decoder.embed_positions = FSDP(model.model.decoder.embed_positions)
    model.lm_head = FSDP(model.lm_head)
    for i in range(len(model.model.decoder.layers)):
        model.model.decoder.layers[i].self_attn.k_proj = FSDP(model.model.decoder.layers[i].self_attn.k_proj)
        model.model.decoder.layers[i].self_attn.q_proj = FSDP(model.model.decoder.layers[i].self_attn.q_proj)
        model.model.decoder.layers[i].self_attn.v_proj = FSDP(model.model.decoder.layers[i].self_attn.v_proj)
        model.model.decoder.layers[i].self_attn.out_proj = FSDP(model.model.decoder.layers[i].self_attn.out_proj)
        model.model.decoder.layers[i].fc1 = Linear(hidden_size, hidden_size * 4, slice_size, slice_size)
        model.model.decoder.layers[i].fc2 = Linear(4 * hidden_size, hidden_size, slice_size, slice_size)
    return model
