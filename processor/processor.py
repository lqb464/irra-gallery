import logging
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
                        model.module.eval(), log=True, return_metrics=True)
                else:
                    epoch_metrics = evaluator.eval(
                        model.eval(), log=True, return_metrics=True)
                top1 = epoch_metrics['text_to_image_r1']

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
