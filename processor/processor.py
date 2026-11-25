import logging
import torch
from prettytable import PrettyTable
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def meter_scalar(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return value


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(meter_scalar(total_loss), batch_size)
            for key in ('sdm_loss', 'itc_loss', 'id_loss', 'mlm_loss',
                        'img_acc', 'txt_acc'):
                if key in ret:
                    meters[key].update(meter_scalar(ret[key]), batch_size)
            if 'mlm_acc' in ret:
                meters['mlm_acc'].update(meter_scalar(ret['mlm_acc']), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', meter_scalar(ret['temperature']), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        epoch_metrics = None
        if epoch >= args.eval_after_epoch and epoch % eval_period == 0:
            if get_rank() == 0:
                if args.distributed:
                    epoch_metrics = evaluator.eval(
                        model.module.eval(), log=False, return_metrics=True)
                else:
                    epoch_metrics = evaluator.eval(
                        model.eval(), log=False, return_metrics=True)
                top1 = epoch_metrics['text_to_image_r1']

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

        if get_rank() == 0:
            parts = [f"Epoch {epoch}/{num_epoch}"]
            for key in ("loss", "sdm_loss", "itc_loss", "id_loss", "mlm_loss",
                        "img_acc", "txt_acc", "mlm_acc"):
                if meters[key].count:
                    parts.append(f"{key}={meters[key].avg:.4f}")
            if epoch_metrics is not None:
                metric_table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
                for column in ("R1", "R5", "R10", "mAP", "mINP"):
                    metric_table.custom_format[column] = lambda field, value: f"{value:.3f}"
                metric_table.add_row([
                    "t2i-base",
                    epoch_metrics["base_text_to_image_r1"],
                    epoch_metrics.get("base_text_to_image_r5", epoch_metrics["text_to_image_r5"]),
                    epoch_metrics.get("base_text_to_image_r10", epoch_metrics["text_to_image_r10"]),
                    epoch_metrics.get("base_text_to_image_mAP", epoch_metrics["text_to_image_mAP"]),
                    epoch_metrics.get("base_text_to_image_mINP", epoch_metrics["text_to_image_mINP"]),
                ])
                if getattr(model.module if args.distributed else model,
                           "gallery_context_enabled", False):
                    metric_table.add_row([
                        "t2i-hard-contrastive",
                        epoch_metrics["text_to_image_r1"],
                        epoch_metrics["text_to_image_r5"],
                        epoch_metrics["text_to_image_r10"],
                        epoch_metrics["text_to_image_mAP"],
                        epoch_metrics["text_to_image_mINP"],
                    ])
                logger.info(" | ".join(parts) + "\n" + str(metric_table))
            else:
                logger.info(" | ".join(parts))


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
