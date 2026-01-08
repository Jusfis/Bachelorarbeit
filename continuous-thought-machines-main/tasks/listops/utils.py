import os
import re
import math
from models.ctm_kan_efficient import ContinuousThoughtMachine
from models.lstm import LSTMBaseline # alternative baseline model

def prepare_model(prediction_reshaper, args, device):
    if args.model_type == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            postactivation_production=args.postactivation_production,
        ).to(device) # below baseline model # todo add other baseline model cnn for instance
    # elif args.model_type == 'lstm':
    #     model = LSTMBaseline(
    #         iterations=args.iterations,
    #         d_model=args.d_model,
    #         num_layers=1,
    #         d_input=args.d_input,
    #         heads=args.heads,
    #         backbone_type=args.backbone_type,
    #         positional_embedding_type=args.positional_embedding_type,
    #         out_dims=args.out_dims,
    #         prediction_reshaper=prediction_reshaper,
    #         dropout=args.dropout,
    #     ).to(device)
    else:
        raise ValueError(f"Model must be either ctm or baseline, not {args.model_type}")

    return model
