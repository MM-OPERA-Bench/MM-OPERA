# MM-OPERA

For Linux:

```bash
export AIGPTX_API_KEY="YOUR_ACTUAL_AIGPTX_KEY_HERE"
```

For Windows:

```bash
$ENV:AIGPTX_API_KEY = "YOUR_ACTUAL_AIGPTX_KEY_HERE"
$ENV:AIGPTX_API_KEY = "sk-Z5VdcIL18691422be52ET3BlBkFJ9bF56d579a444dc79179"
```


```bash
python -m evaluation.RIA.RIA_run --model_name Gemini-2.0-Flash-Thinking-Exp
```

```bash
python -m evaluation.RIA.RIA_regular_judge --test_model_name Gemini-2.0-Flash-Thinking-Exp --judge_model_name GPT-4o-judge
```

```bash
python -m evaluation.RIA.RIA_reasoning_judge --test_model_name Gemini-2.0-Flash-Thinking-Exp --judge_model_name GPT-4o-judge
```
