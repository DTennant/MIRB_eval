import transformers
import os
import torch
import sys

def load_i2t_model(engine, args=None):
    if engine == 'otter-mpt':
        from otter_ai import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="cuda", torch_dtype=torch.bfloat16)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'otter-llama':
        from otter_ai import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-LLaMA7B-LA-InContext", device_map="cuda", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'llava15-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.5-7b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'llava16-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-7b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'llava16-13b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-13b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'phi3-vision':
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
        processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        tokenizer = processor.tokenizer
        processor.image_processor.num_crops=1
    elif 'vila' in engine:
        import sys
        sys.path.append('/mnt/ceph_rbd/multimodal/long-vllm/VILA')
        from llava.model.builder import load_pretrained_model as load_llava_model
        model_path = 'Efficient-Large-Model/VILA-7B' if 'vila-7b' in engine else 'Efficient-Large-Model/VILA-2.7b'
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(model_path)
        
        tokenizer, model, image_processor, context_len = load_llava_model(
                model_path,
                model_base=None, model_name="VILA", 
                attn_implementation="flash_attention_2",
                device_map="auto", 
                config=config,
            )
        processor = image_processor
        #model.config.use_cache = False
        # if args.extend:
        #     window_size = 1024
        #     group_size = 16
        #     max_length = (4096 - window_size) * group_size + window_size
        #     tokenizer.model_max_length = max_length
        #     model.config.max_position_embeddings = max_length
        #     model.config.tokenizer_model_max_length = max_length
        #     SelfExtend.apply(model, group_size=group_size, window_size=window_size, enable_flash_attention=True, flash_attention_impl="flash_attn")
    elif engine == 'qwen-vl-chat':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", 
                                                                  trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        processor = None
    elif engine == 'qwen-vl':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        processor = None
    elif engine == 'internlm-x2':
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    elif engine == 'internlm-x2-hd':
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    elif engine == 'openflamingo':
        from open_flamingo import create_model_and_transforms
        model, processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        model = model.to(torch.bfloat16).cuda()

    elif engine == 'emu2-chat':
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2-Chat",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        # adjust according to your device
        device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',2:'38GiB',3:'38GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])
        device_map["model.decoder.lm.lm_head"] = 0

        model = load_checkpoint_and_dispatch(
            model, 
            '/mnt/ceph_rbd/huggingface/models--BAAI--Emu2-Chat/snapshots/20ea30b04f8fee599cf97535e655c200df728501',
            device_map=device_map).eval()
        processor = None
    elif engine == 'idefics1-9b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics1-9b':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics1-80b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics2-8b':
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
        model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16, 
                                                                 device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer = processor.tokenizer
    elif 'gpt4v' in engine or 'gemini' in engine:
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
    return model, tokenizer, processor

def load_t2i_model(engine, args):
    if engine == 'emu2-gen':
        # git clone https://github.com/baaivision/Emu/tree/main
        from Emu.Emu2.emu.diffusion import EmuVisualGeneration
        # git clone https://huggingface.co/BAAI/Emu2-Gen
        # set path to this folder
        path = "path/to/Emu2-Gen"
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"{path}/tokenizer")

        # download model weigths from https://model.baai.ac.cn/model-detail/220122 
        pipe = EmuVisualGeneration.from_pretrained(
            'path/to/Emu2-Gen_pytorch_model.bf16.safetensors',
            dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        model = pipe
        processor = None
        model.multito(["cuda:0", "cuda:1",])
    elif engine == 'emu1-gen':
        # git clone https://github.com/baaivision/Emu/tree/main
        # git clone https://huggingface.co/BAAI/Emu
        from Emu.Emu1.models.modeling_emu import Emu
        from Emu.Emu1.models.pipeline import EmuGenerationPipeline
        import sys
        sys.path.append('path/to/Emu/Emu1')
        args = type('Args', (), {
            "instruct": False,
            "ckpt_path": 'path/to/Emu/pretrain', # huggingface weights
            "device": torch.device('cuda'),
        })()

        model = EmuGenerationPipeline.from_pretrained(
            path=args.ckpt_path,
            args=args,
        )
        tokenizer, processor = None, None
    elif engine == 'gill':
        from gill.gill.models import load_gill
        import sys
        # git clone https://github.com/kohjingyu/gill
        sys.path.append('path/to/gill')
        model_dir = 'path/to/gill/checkpoints/gill_opt'
        model = load_gill(model_dir, load_ret_embs=False)
        model = model.cuda()
        tokenizer, processor = None, None
    elif 'seed-llama' in engine:
        from omegaconf import OmegaConf
        import hydra, sys
        # git clone https://github.com/AILab-CVC/SEED
        os.environ['PROJECT_ROOT'] = 'path/to/SEED/'
        sys.path.append('path/to/SEED')
        from models.model_tools import get_pretrained_llama_causal_model
        from models import seed_llama_tokenizer
        tokenizer_cfg_path = f'path/to/SEED/configs/tokenizer/seed_llama_tokenizer_hf.yaml'
        tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
        tokenizer = hydra.utils.instantiate(
            tokenizer_cfg, device='cuda', load_diffusion=True)

        transform_cfg_path = f'path/to/SEED/configs/transform/clip_transform.yaml'
        transform_cfg = OmegaConf.load(transform_cfg_path)
        transform = hydra.utils.instantiate(transform_cfg)

        model_size = engine.split('-')[-1]
        model_cfg = OmegaConf.load(f'path/to/SEED/configs/llm/seed_llama_{model_size}.yaml')
        model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        processor = transform

    return model, tokenizer, processor