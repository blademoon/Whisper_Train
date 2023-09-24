# Вспомогательные библиотеки
import os
import subprocess
import os
import multiprocessing
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import datetime
from time import sleep



# HuggingFace
from huggingface_hub import notebook_login, login
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers.utils import logging

# Evaluate
import evaluate
# PyTorch
import torch
print("Required libraries/modules initialization complete!")

# Время начала и конца дешевого тарифа для трехтарифного счётчика
start_time = "23:00:00"
end_time = "07:00:00"
login(token="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")
common_voice =  load_dataset("artyomboyko/common_voice_13_0_ru_dataset_for_whisper_fine_tune")
reference_model = "openai/whisper-small"
result_model_name = "artyomboyko/whisper-small-fine_tuned-ru"
tokenizer_language = "Russian"
model_task = "transcribe"

feature_extractor = WhisperFeatureExtractor.from_pretrained(reference_model)
print("The feature extractor is loaded successfully.")

tokenizer = WhisperTokenizer.from_pretrained(reference_model,
                                             language = tokenizer_language,
                                             task = model_task)
print("Tokenizer loaded successfully.")

processor = WhisperProcessor.from_pretrained(reference_model,
                                             language = tokenizer_language,
                                             task = model_task)
print("Processor loaded successfully.")


def operates_during_cheap_electricity_tariffs(start_time, end_time):

    # Функция для проверки укладывается ли текущее время в диапазон дешевого тарифа
    def in_between(now, start, end):
        if start <= end:
            return start <= now < end
        else: # over midnight e.g., 23:30-04:15
            return start <= now or now < end

    # Функций возвращающая текущее время timedelta
    def get_now():
        now = (datetime.datetime.now()).time()
        td_now = datetime.timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
        return td_now

    # Функция конвертирующая время суток в 24-часово формате в timedelta
    def convert_str_timedelta(time):
        result = (datetime.datetime.strptime(time, "%H:%M:%S")).time() 
        result = datetime.timedelta(hours=result.hour, minutes=result.minute, seconds=result.second)
        return result

    td1 = convert_str_timedelta(start_time)
    td2 = convert_str_timedelta(end_time)
    td_now = get_now()

    if not in_between(td_now, td1, td2):
        time_wait = (td1 - td_now)
        print("Current time:", td_now ," Work has been suspended until the time of the cheaper electricity tariff at", start_time)
        sleep(time_wait.total_seconds())

        td_now = get_now()
        print("Current time:", td_now ,"Work resumed.")

print("Auxiliary functions defined.")




@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # разделить входы и метки, так как они должны быть разной длины и нуждаться в разных методах дополнения (padding)
        # сначала обрабатываем аудиовходы, просто возвращая тензоры torch
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # получим токенизированные последовательности меток
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # заменим дополнение (padding) на -100, чтобы корректно игнорировать потери
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # если токен bos был добавлен на предыдущем шаге токенизации,
        # вырезаем здесь токен bos, так как он все равно будет добавлен позже
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


metric = evaluate.load("wer")
#metric_cer = evaluate.load("cer")

print("Metric is loaded successfully.")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # заменяем -100 на pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # мы не хотим группировать токены при вычислении метрики
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    #cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    operates_during_cheap_electricity_tariffs(start_time, end_time)

    #return {"wer": wer, "cer": cer}
    return {"wer": wer}


print("The compute_metrics function was defined successfully.")


print("Current reference model:", reference_model)
model = WhisperForConditionalGeneration.from_pretrained(reference_model)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir = result_model_name,      # измените имя репозитория по своему усмотрению
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,       # увеличивается в 2x раза при каждом уменьшении размера батча в 2x раза  
    learning_rate=1e-6,
    warmup_steps=250,
    max_steps=100000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    optim = "adamw_torch",
    per_device_eval_batch_size=8,        # Самый простой способ задать равным параметру per_device_train_batch_size
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(

    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Current model: ", reference_model)
print("Current dataset: ")
#trainer.train()
trainer.train(resume_from_checkpoint=True)

print("Training of the model is complete. Push the model, preprocessor and tokeniser into the hub.")
trainer.push_to_hub()
tokenizer.push_to_hub(result_model_name)
processor.push_to_hub(result_model_name)

print("The training has been successfully completed! Check the result in the hub.")
