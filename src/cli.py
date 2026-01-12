from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from logging import getLogger as get_logger


_logger = get_logger(__name__)


def create_gpt_config_from_args(args: Namespace):
    from src.model.config import GptConfig
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


def run_download(args: Namespace) -> None:
    from src.scripts.download import run_download_flow
    run_download_flow(args.sizes, args.dir, args.convert)


def run_pretrain(args: Namespace) -> None:
    from src.scripts.train import run_model_training_flow
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
    from src.scripts.generate import run_model_generation_flow, run_model_interactive_flow
    if args.prompt is not None:
        run_model_generation_flow(
            model_path=args.model_path,
            prompt=args.prompt,
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
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device_type=args.device,
            seed=args.seed
        )


def run_finetune_classification(args: Namespace) -> None:
    from src.scripts.finetune.classification import run_classification_finetuning_flow
    run_classification_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        tuning_set_path=args.tuning_set_path,
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
    from src.scripts.finetune.instruction import run_instruction_finetuning_flow
    run_instruction_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
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
        test_output_path=args.test_output_path,
        evaluate=args.evaluate
    )


def run_finetune_instruction_advanced(args: Namespace) -> None:
    from src.scripts.finetune.instruction_adv import run_instruction_finetuning_advanced_flow
    _logger.warning("Running advanced instruction fine-tuning ")
    run_instruction_finetuning_advanced_flow(
        pretrained_model_path=args.pretrained_model_path,
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


def run_finetune_coding(args: Namespace) -> None:
    from src.scripts.finetune.code_instruction import run_coding_finetuning_flow
    run_coding_finetuning_flow(
        pretrained_model_path=args.pretrained_model_path,
        dataset_path=args.dataset_path,
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
        test_output_path=args.test_output_path,
        evaluate=args.evaluate,
        max_samples=args.max_samples
    )


def run_spam_ham(args: Namespace) -> None:
    from src.scripts.classify import run_spam_ham_flow, run_spam_ham_interactive_flow
    if args.text is not None:
        run_spam_ham_flow(
            model_path=args.model_path,
            text=args.text,
            device_type=args.device,
            seed=args.seed
        )
    else:
        run_spam_ham_interactive_flow(
            model_path=args.model_path,
            device_type=args.device,
            seed=args.seed
        )


def run_chat(args: Namespace) -> None:
    from src.scripts.chat import run_chat_flow
    run_chat_flow(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_type=args.device,
        seed=args.seed
    )


def run_coder(args: Namespace) -> None:
    from src.scripts.coder import run_coder_flow
    run_coder_flow(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_type=args.device,
        seed=args.seed
    )


def cli() -> None:
    parser = ArgumentParser(description="LLM from Scratch - Main CLI", formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = False

    # Add minimal subcommand stubs without importing anything
    subparsers.add_parser("download", help="Download GPT2 model files", add_help=False)
    subparsers.add_parser("pretrain", help="Train a GPT2 foundation model from scratch", add_help=False)
    subparsers.add_parser("generate", help="Generate text using a pre-trained GPT2 model", add_help=False)
    subparsers.add_parser("finetune", help="Fine-tune a pre-trained GPT2 foundation model", add_help=False)
    subparsers.add_parser("spam-ham", help="Classify text as spam or ham using a classification fine-tuned model", add_help=False)
    subparsers.add_parser("chat", help="Chat with an instruction fine-tuned assistant model", add_help=False)
    subparsers.add_parser("coder", help="Interactive coding session with a code fine-tuned model", add_help=False)

    # Quick parse to see if we're just showing help
    args, remaining = parser.parse_known_args()
    if not args.command or '--help' in remaining or '-h' in remaining:
        if not args.command:
            parser.print_help()
            return


    # Rebuild parser with full details for the specific command
    parser = ArgumentParser(description="LLM from Scratch - Main CLI", formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    if args.command == "download":
        from src.scripts.download import add_arguments as add_download_arguments
        download_parser = subparsers.add_parser(
            "download",
            help="Download GPT-2 model files",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_download_arguments(download_parser)
        download_parser.set_defaults(func=run_download)

    elif args.command == "pretrain":
        from src.scripts.train import add_arguments as add_pretrain_arguments
        from src.model.config import add_arguments as add_gpt_config_arguments
        pretrain_parser = subparsers.add_parser(
            "pretrain",
            help="Train a GPT model from scratch",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_gpt_config_arguments(pretrain_parser)
        add_pretrain_arguments(pretrain_parser)
        pretrain_parser.set_defaults(func=run_pretrain)

    elif args.command == "generate":
        from src.scripts.generate import add_arguments as add_generate_arguments
        generate_parser = subparsers.add_parser(
            "generate",
            help="Generate text using a pre-trained GPT2 model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_generate_arguments(generate_parser)
        generate_parser.set_defaults(func=run_generate)

    elif args.command == "finetune":
        from src.scripts.finetune.classification import add_arguments as add_classification_arguments
        from src.scripts.finetune.instruction import add_arguments as add_instruction_arguments
        from src.scripts.finetune.instruction_adv import add_arguments as add_instruction_advanced_arguments
        from src.scripts.finetune.code_instruction import add_arguments as add_coding_arguments

        finetune_parser = subparsers.add_parser(
            "finetune",
            help="Fine-tune a pre-trained GPT2 model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        finetune_subparsers = finetune_parser.add_subparsers(dest="finetune_type", help="Fine-tuning type")
        finetune_subparsers.required = True

        # Classification
        classification_parser = finetune_subparsers.add_parser(
            "classification",
            help="Fine-tune for classification tasks",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_classification_arguments(classification_parser)
        classification_parser.set_defaults(func=run_finetune_classification)

        # Instruction (Basic)
        instruction_parser = finetune_subparsers.add_parser(
            "instruction",
            help="Fine-tune for instruction following",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_instruction_arguments(instruction_parser)
        instruction_parser.set_defaults(func=run_finetune_instruction)

        # Instruction (Advanced)
        instruction_advanced_parser = finetune_subparsers.add_parser(
            "instruction-adv",
            help="Fine-tune for instruction following with advanced features (LoRA, masking, Phi-3, Alpaca52k)",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_instruction_advanced_arguments(instruction_advanced_parser)
        instruction_advanced_parser.set_defaults(func=run_finetune_instruction_advanced)

        # Coding (Personal Project)
        coding_parser = finetune_subparsers.add_parser(
            "coding",
            help="Fine-tune specifically for Python code generation with loss masking",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_coding_arguments(coding_parser)
        coding_parser.set_defaults(func=run_finetune_coding)

    elif args.command == "spam-ham":
        from src.scripts.classify import add_arguments as add_spam_ham_arguments
        spam_ham_parser = subparsers.add_parser(
            "spam-ham",
            help="Classify text as spam or ham using a fine-tuned classification model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_spam_ham_arguments(spam_ham_parser)
        spam_ham_parser.set_defaults(func=run_spam_ham)

    elif args.command == "chat":
        from src.scripts.chat import add_arguments as add_chat_arguments
        chat_parser = subparsers.add_parser(
            "chat",
            help="Chat with an instruction fine-tuned assistant model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_chat_arguments(chat_parser)
        chat_parser.set_defaults(func=run_chat)

    elif args.command == "coder":
        from src.scripts.coder import add_arguments as add_coder_arguments
        coder_parser = subparsers.add_parser(
            "coder",
            help="Interactive coding session with a code fine-tuned model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        add_coder_arguments(coder_parser)
        coder_parser.set_defaults(func=run_coder)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
