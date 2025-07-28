import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb


class PretrainLogger:
    def __init__(self, trial_name, fold, use_wandb=False):
        self.trial_name = trial_name
        self.fold = fold
        self.use_wandb = use_wandb
        self.logger = self._setup_logging()
        self.writer = self._setup_tensorboard()

        if use_wandb:
            wandb.init(project="solar_flare_pretrain", name=trial_name)

    def _setup_logging(self):
        """Setup logging"""
        log_dir = os.path.join("logs", "pretrain", self.trial_name, f"fold{self.fold}")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            log_dir, f"{self.trial_name}_fold{self.fold}_{timestamp}.log"
        )

        logger = logging.getLogger("pretrain_log")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _setup_tensorboard(self):
        """Setup TensorBoard"""
        return SummaryWriter(os.path.join("logs", "pretrain", self.trial_name))

    def log_config(self, args):
        """Log configuration information"""
        self.logger.info("=== Experiment Configuration ===")
        self.logger.info(f"Trial Name: {args.trial_name}")
        self.logger.info(f"Fold: {args.fold}")
        self.logger.info(f"Mode: {args.mode}")
        self.logger.info(f"Batch Size: {args.batch_size}")
        self.logger.info(f"Mask Ratio: {args.mask_ratio}")
        self.logger.info(f"CUDA Device: {args.cuda_device}")
        self.logger.info(f"Input Directory: {args.input_dir}")
        self.logger.info(f"Output Directory: {args.output_dir}")
        self.logger.info(f"Data Root: {args.data_root}")
        self.logger.info("============================")

    def log_train_step(self, epoch, train_loss, val_metrics, lr, is_best=False):
        """Log training step"""
        log_message = f"\nEpoch {epoch}:"
        log_message += f"\n  Training Loss: {train_loss:.6f}"
        log_message += f"\n  Validation Metrics:"
        log_message += f"\n    Full MSE: {val_metrics['mse']:.6f}"
        log_message += f"\n    Full MAE: {val_metrics['mae']:.6f}"
        log_message += f"\n    Solar MSE: {val_metrics['solar_mse']:.6f}"
        log_message += f"\n    Solar MAE: {val_metrics['solar_mae']:.6f}"
        log_message += f"\n  Learning Rate: {lr:.6f}"

        if is_best:
            log_message += (
                f"\n  New best model saved! (Solar MSE: {val_metrics['solar_mse']:.6f})"
            )

        self.logger.info(log_message)

        # TensorBoard
        self.writer.add_scalar("train_loss", train_loss, global_step=epoch)
        self.writer.add_scalar("val/mse", val_metrics["mse"], global_step=epoch)
        self.writer.add_scalar("val/mae", val_metrics["mae"], global_step=epoch)
        self.writer.add_scalar(
            "val/solar_mse", val_metrics["solar_mse"], global_step=epoch
        )
        self.writer.add_scalar(
            "val/solar_mae", val_metrics["solar_mae"], global_step=epoch
        )

        # wandb
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val/mse": val_metrics["mse"],
                    "val/mae": val_metrics["mae"],
                    "val/solar_mse": val_metrics["solar_mse"],
                    "val/solar_mae": val_metrics["solar_mae"],
                    "learning_rate": lr,
                }
            )

    def log_final_metrics(self, test_metrics):
        """Log final test evaluation"""
        final_log = "\nFinal Test Metrics (Best Model):"
        final_log += f"\n  Full MSE: {test_metrics['mse']:.6f}"
        final_log += f"\n  Full MAE: {test_metrics['mae']:.6f}"
        final_log += f"\n  Solar MSE: {test_metrics['solar_mse']:.6f}"
        final_log += f"\n  Solar MAE: {test_metrics['solar_mae']:.6f}"

        self.logger.info(final_log)

        if self.use_wandb:
            wandb.run.summary.update(
                {
                    "test/mse": test_metrics["mse"],
                    "test/mae": test_metrics["mae"],
                    "test/solar_mse": test_metrics["solar_mse"],
                    "test/solar_mae": test_metrics["solar_mae"],
                }
            )

    def log_model_summary(self, model_summary):
        """Log model summary"""
        self.logger.info(f"Model Summary:\n{model_summary}")

    def log_info(self, message):
        """Log general information"""
        self.logger.info(message)

    def log_warning(self, message):
        """Log warning"""
        self.logger.warning(message)

    def log_error(self, message):
        """Log error"""
        self.logger.error(message)

    def finish(self):
        """Finish logging"""
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
