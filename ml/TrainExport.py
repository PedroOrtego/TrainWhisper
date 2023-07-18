import sys
import pandas as pd
import whisper
from datasets import DatasetDict, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

def load_data(dataset_name):
    """
    Load the dataset.

    Returns:
        dataset: dataset.
    """
    df = pd.read_csv(dataset_name)


    dataset = DatasetDict()

    # split the df into train and test sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # load the train and test sets into the dataset
    dataset['train'] = Dataset.from_pandas(train_df)
    dataset['test'] = Dataset.from_pandas(test_df)

    # remove the columns that are not needed
    dataset = dataset.remove_columns(['audio_clipping', 'audio_clipping:confidence', 'background_noise_audible', 'background_noise_audible:confidence', 'overall_quality_of_the_audio', 'quiet_speaker', 'quiet_speaker:confidence', 'speaker_id', 'file_download', 'prompt', 'writer_id'])

    # Create a new column called audio, this will be a dictionary containing the path (from the dataset column file_name), the array (from the whisper.load_audio function) and the sampling rate (from the whisper.load_audio function)
    dataset = dataset.map(lambda x: {'audio': {'path': '/valohai/inputs/dataset/data/' + x['file_name'], 'array': whisper.load_audio('/valohai/inputs/dataset/data/' + x['file_name']), 'sampling_rate': 16000}})


    # Drop the file_name column
    dataset = dataset.remove_columns(['file_name'])

    # Change the name of the column phrase to sentence
    dataset = dataset.rename_column('phrase', 'sentence')

    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python TrainExport.py <dataset> <whisper_model_name> <output_model_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    whisper_model_name = "openai/" + sys.argv[2]
    output_model_name = sys.argv[3]

    dataset = load_data(dataset_name)

    def prepare_dataset(batch):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name, language="English", task="transcribe")
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2)

    processor = WhisperProcessor.from_pretrained(whisper_model_name, language="English", task="transcribe")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name, language="English", task="transcribe")

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./" + output_model_name,  # Repo name of the model
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()