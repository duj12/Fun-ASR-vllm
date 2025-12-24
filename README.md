# Fun-ASR



Fun-ASR 



# Environment Setup 🐍

```shell
git clone https://github.com/yuekaizhang/Fun-ASR-vllm.git
cd Fun-ASR-vllm
apt-get install -y ffmpeg
uv pip install -r requirements.txt
```

<a name="usage-tutorial"></a>

# TODO

- [ ] Support encoder TensorRT
- [ ] Support batch > 1 Inference
- [ ] Support Nvidia Triton Inference Server

# Usage 🛠️

## Inference

### Direct Inference

```python
from model import FunASRNano
from vllm import LLM, SamplingParams

def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    vllm = LLM(model="yuekai/Fun-ASR-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(
        top_p=0.001,
        max_tokens=500,
    )
    m.vllm = vllm
    m.vllm_sampling_params = sampling_params

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```


### Benchmark Test

```bash
dataset_name="yuekai/speechio"
subset_name="SPEECHIO_ASR_ZH00007"
split_name="test"

uv run python \
    infer.py \
    --model_dir FunAudioLLM/Fun-ASR-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 1 \
    --log_dir ./logs_vllm_test2_$dataset_name_$subset_name \
    --vllm_model_dir yuekai/Fun-ASR-Nano-2512-vllm
```

