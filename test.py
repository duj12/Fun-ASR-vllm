from model import FunASRNano
from vllm import LLM, SamplingParams

def main(args):
    model_dir = args.model_dir
    # Load the base model
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device=args.device)
    m.eval()
    
    # Initialize vLLM
    if args.enable_vllm:
        vllm = LLM(model="yuekai/Fun-ASR-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=args.gpu_memory_utilization)
        sampling_params = SamplingParams(
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        
        # Attach vLLM to the model
        m.vllm = vllm
        m.vllm_sampling_params = sampling_params

    # Run inference
    wav_path = args.audio_in
    kwargs['itn']= args.itn == 1 
    print(f"itn: {kwargs['itn']}")
    res = m.inference(data_in=[wav_path], **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("--model_dir", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512", help="Path to the model directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens for vLLM sampling")
    parser.add_argument("--top_p", type=float, default=0.001, help="Top-p sampling parameter for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4, help="GPU memory utilization for vLLM")
    parser.add_argument("--enable_vllm", action="store_true", help="Enable vLLM for inference")
    parser.add_argument("--audio_in", type=str, default=None, help="Path to the input wav file")
    parser.add_argument("--itn", type=int, default=0, help="Enable ITN post-processing")
    
    args = parser.parse_args()
    main(args)