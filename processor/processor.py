import logging
import time
import torch
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
    device = next(model.parameters()).device
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
        "mlm_acc": AverageMeter(),
        "hard_contrastive_loss": AverageMeter(),
        "gallery_candidate_count": AverageMeter(),
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
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(meter_scalar(ret.get('sdm_loss', 0)), batch_size)
            meters['itc_loss'].update(meter_scalar(ret.get('itc_loss', 0)), batch_size)
            meters['id_loss'].update(meter_scalar(ret.get('id_loss', 0)), batch_size)
            meters['mlm_loss'].update(meter_scalar(ret.get('mlm_loss', 0)), batch_size)
            meters['hard_contrastive_loss'].update(
                meter_scalar(ret.get('hard_contrastive_loss', 0)), batch_size
            )
            meters['gallery_candidate_count'].update(
                meter_scalar(ret.get('gallery_candidate_count', 0)), batch_size
            )

            meters['img_acc'].update(meter_scalar(ret.get('img_acc', 0)), batch_size)
            meters['txt_acc'].update(meter_scalar(ret.get('txt_acc', 0)), batch_size)
            meters['mlm_acc'].update(meter_scalar(ret.get('mlm_acc', 0)), 1)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            synchronize()

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', meter_scalar(ret['temperature']), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if epoch >= args.eval_after_epoch and epoch % eval_period == 0:
            if get_rank() == 0:
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

        if get_rank() == 0:
            parts = [f"Epoch {epoch}/{num_epoch}"]
            for key, meter in meters.items():
                if meter.count and (meter.avg > 0 or key.startswith("gallery_delta_")):
                    parts.append(f"{key}={meter.avg:.4f}")
            logger.info(" | ".join(parts))


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())

