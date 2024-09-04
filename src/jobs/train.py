from typing import List, Dict, Literal
from datasets import Dataset # type: ignore
from peft import LoraConfig, AutoPeftModelForCausalLM # type: ignore
from transformers import ( # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
from functools import cached_property
from pydantic import BaseModel, Field, computed_field
from ..shared.utils import ttl_cache






@ttl_cache()
def load_models(*, llm_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        llm_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(
        llm_id, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class LlamaFineTuner(BaseModel):
    llm_id: Literal["meta-llama/Meta-Llama-3.1-8B-Instruct"] = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    suffix: str = Field(...)
    training_data: List[Dict[str, str]] = Field(default_factory=list)
    @computed_field(return_type=str)
    @cached_property
    def output_model(self) -> str:
        return f"obahamonde/Meta-Llama-3.1-{self.suffix}"

    @cached_property
    def lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    @cached_property
    def training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_model,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            remove_unused_columns=False,
        )

    @cached_property
    def trainer(self) -> SFTTrainer:
        model, tokenizer = load_models(llm_id=self.llm_id)

        # Tokenize the dataset
        def tokenize_function(examples):
            # Ensure that examples["text"] is a string or list of strings
            return tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=1024
            )

        tokenized_dataset = Dataset.from_list(self.training_data).map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return SFTTrainer(
            model=model,
            args=self.training_arguments,
            train_dataset=tokenized_dataset,
            peft_config=self.lora_config,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
            dataset_text_field="text",
        )

    def fine_tune(self, training_data: List[Dict[str, str]]):
        self.training_data = training_data  # Store training data in the object
        trainer = self.trainer  # This will now correctly access training data
        trainer.train()

    def generate_response(self, user_input: str) -> str:
        model, tokenizer = load_models(llm_id=self.llm_id)
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant:"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample=True,
            top_k=5,
            temperature=0.5,
            repetition_penalty=1.2,
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
        )

        start_time = perf_counter()
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = perf_counter() - start_time

        print(f"Time taken for inference: {round(inference_time, 2)} seconds")
        return response


@ttl_cache()
def create_fine_tuner(suffix: str) -> LlamaFineTuner:
    return LlamaFineTuner(suffix=suffix)

`

def main():
    