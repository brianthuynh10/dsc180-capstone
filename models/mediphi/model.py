import pandas as pd
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

class MediPhiModel: 
    def __init__(self, fine_tuned=True): 
        base_model_id = "microsoft/MediPhi"
        adapter_model_id = "brianthuynh/mediphi-LoRA-Edema" # Your HF repo

        if fine_tuned:
            print("Loading fine-tuned MediPhi model with LoRA adapter...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.model = PeftModel.from_pretrained(self.model, adapter_model_id)
        else:
            print("Loading base MediPhi model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    def _ask_llm_batch(self, reports, prompt):
        messages = [
        [{"role": "user", "content": prompt + report}]
        for report in reports
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
            )

        decoded = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # CLEAR UNUSED CUDA MEMORY
        del inputs, outputs
        torch.cuda.empty_cache()

        return [(d) for d in decoded]
    
    
    def make_predictions(self, prompt, df: pd.DataFrame, BATCH_SIZE=10):
        """
        Make predictions on a DataFrame column[ReportClean] of reports using the LLM in batches. 
        """
        results = []

        total = len(df)
        for i in tqdm(
            range(0, total, BATCH_SIZE),
            desc="Running LLM inference",
            unit="reports",
        ):
            batch_reports = df["ReportClean"].iloc[i:i + BATCH_SIZE].tolist()
            batch_results = self._ask_llm_batch(batch_reports, prompt)
            results.extend(batch_results)

        return results
    
