*For Multi Lingual Donwlaod*

```hf download convaiinnovations/laya --local-dir ./models```

---

*Only English*

```
hf download convaiinnovations/laya \
  --local-dir ./models/laya-english \
  --include "model.safetensors" \
  --include "rl_agent_config.json" \
  --include "encoder/*" \
  --include "tokenizer/*"
```


Source: https://github.com/NandhaKishorM/laya
