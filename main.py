from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from logging import getLogger as get_logger

from src.model.config import GptConfig, add_arguments as add_gpt_config_arguments
from src.scripts.download import download_gpt2, add_arguments as add_download_arguments
from src.scripts.evaluate import run_ollama_evaluation_flow, add_arguments as add_evaluate_arguments
from src.scripts.finetune.classification import run_classification_finetuning_flow, add_arguments as add_classification_arguments
from src.scripts.finetune.instruction import run_instruction_finetuning_flow, add_arguments as add_instruction_arguments
from src.scripts.finetune.instruction_advanced import run_instruction_finetuning_advanced_flow, add_arguments as add_instruction_advanced_arguments
from src.scripts.generate import run_model_generation_flow, run_model_interactive_flow, add_arguments as add_generate_arguments
from src.scripts.pretrain import run_model_training_flow, add_arguments as add_pretrain_arguments


_logger = get_logger(__name__)


def create_gpt_config_from_args(args: Namespace) -> GptConfig:
    return GptConfig(
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        qkv_bias=args.use_qkv_bias,
        kv_window_size=args.kv_window_size
    )


def run_pretrain(args: Namespace) -> None:
    config = create_gpt_config_from_args(args)
    run_model_training_flow(
        config=config,
        training_set_path=args.training_set_path,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        dataset_encoding=args.dataset_encoding,
        device_type=args.device,
        seed=args.seed,
        max_length=args.max_length,
        stride=args.stride,
        train_ratio=args.train_ratio,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        saved_model_path=args.saved_model_path,
        saved_plot_path=args.saved_plot_path
    )


def run_generate(args: Namespace) -> None:
    config = create_gpt_config_from_args(args)
    if args.prompt is not None:
        run_model_generation_flow(
            config=config,
            prompt=args.prompt,
            models_dir=args.models_dir,
            model_size=args.model_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device_type=args.device,
            seed=args.seed,
            measure_time=args.measure_time,
            measure_memory=args.measure_memory
        )
    else:
        run_model_interactive_flow(
            config=config,
            models_dir=args.models_dir,
            model_size=args.model_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device_type=args.device,
            seed=args.seed
        )


def run_finetune_classification(args: Namespace) -> None:
    config = create_gpt_config_from_args(args)
    run_classification_finetuning_flow(
        config=config,
        models_dir=args.models_dir,
        model_size=args.model_size,
        tuning_set_path=args.tuning_set_path,
        sep=args.sep,
        column_names=args.column_names,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        save_split_dir=args.save_split_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        accuracy_plot_save_path=args.accuracy_plot_save_path,
        model_save_path=args.model_save_path
    )


def run_finetune_instruction(args: Namespace) -> None:
    config = create_gpt_config_from_args(args)
    run_instruction_finetuning_flow(
        config=config,
        models_dir=args.models_dir,
        model_size=args.model_size,
        tuning_set_path=args.tuning_set_path,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        model_save_path=args.model_save_path,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=args.pad_token_id,
        test_output_path=args.test_output_path
    )


def run_finetune_instruction_advanced(args: Namespace) -> None:
    _logger.warning("Running advanced instruction fine-tuning ")
    config = create_gpt_config_from_args(args)
    run_instruction_finetuning_advanced_flow(
        config=config,
        models_dir=args.models_dir,
        model_size=args.model_size,
        tuning_set_path=args.tuning_set_path,
        use_alpaca52k=args.use_alpaca52k,
        mask_instructions=args.mask_instructions,
        use_phi3_prompt=args.use_phi3_prompt,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        device_type=args.device,
        lr=args.lr,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        loss_plot_save_path=args.loss_plot_save_path,
        model_save_path=args.model_save_path,
        max_new_tokens=args.max_new_tokens,
        test_output_path=args.test_output_path
    )


def run_download(args: Namespace) -> None:
    download_gpt2(args.model_size, args.models_dir)


def run_evaluate(args: Namespace) -> None:
    run_ollama_evaluation_flow(args.file_path, args.model)


def main() -> None:
    parser = ArgumentParser(description="LLM from Scratch - Main CLI", formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Pretrain command
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help="Train a GPT model from scratch",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(pretrain_parser)
    add_pretrain_arguments(pretrain_parser)
    pretrain_parser.set_defaults(func=run_pretrain)

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text using a pre-trained GPT model",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(generate_parser)
    add_generate_arguments(generate_parser)
    generate_parser.set_defaults(func=run_generate)

    # Finetune command with subcommands
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune a pre-trained model",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    finetune_subparsers = finetune_parser.add_subparsers(dest="finetune_type", help="Fine-tuning type")
    finetune_subparsers.required = True

    # Finetune - Classification
    classification_parser = finetune_subparsers.add_parser(
        "classification",
        help="Fine-tune for classification tasks",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(classification_parser)
    add_classification_arguments(classification_parser)
    classification_parser.set_defaults(func=run_finetune_classification)

    # Finetune - Instruction
    instruction_parser = finetune_subparsers.add_parser(
        "instruction",
        help="Fine-tune for instruction following",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(instruction_parser)
    add_instruction_arguments(instruction_parser)
    instruction_parser.set_defaults(func=run_finetune_instruction)

    # Finetune - Instruction Advanced
    instruction_advanced_parser = finetune_subparsers.add_parser(
        "instruction-advanced",
        help="Fine-tune for instruction following with advanced features (LoRA, masking, Phi-3, Alpaca52k)",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_gpt_config_arguments(instruction_advanced_parser)
    add_instruction_advanced_arguments(instruction_advanced_parser)
    instruction_advanced_parser.set_defaults(func=run_finetune_instruction_advanced)

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download GPT-2 model files",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_download_arguments(download_parser)
    download_parser.set_defaults(func=run_download)

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model responses with Ollama API",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_evaluate_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=run_evaluate)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
