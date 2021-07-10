from models.model_with_TCN_big_new_one_batch_hwdb import Model
from dataset.data_utils_kernel_box import MyDataset, AlignCollate
from models.loss_kernels import DICE_loss
from models.loss_ctc import ctc_loss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
from tqdm import tqdm

from torch.cuda.amp import GradScaler
from dataset.hwdb2_0_chars import char_dict, char_set
import torch.multiprocessing
from torch.cuda.amp import autocast as autocast

torch.multiprocessing.set_sharing_strategy('file_system')

scaler = GradScaler()


def eval(model, evel_dataset, criterion_kernel, criterion_char, epoch, is_save=True):
    eval_dataloader = DataLoader(dataset=evel_dataset, collate_fn=AlignCollate(), batch_size=2, shuffle=True, num_workers=0,
                                 pin_memory=False)
    evel_steps = len(eval_dataloader)
    evel_iter = iter(eval_dataloader)
    model.eval()
    pbar = tqdm(total=evel_steps)
    a_CR_correct_chars, a_AR_correct_chars, a_all_chars = 0, 0, 0
    loss_all = 0
    loss_kernel_all = 0
    loss_char_all = 0
    editdistence = []
    with torch.no_grad():
        for evel_step in range(evel_steps):
            imgs, kernel_labels, text_polys, label_tensors, text_lengths, _ = next(evel_iter)
            # torch.cuda.empty_cache()
            imgs = imgs.to(device)
            kernel_labels = kernel_labels.to(device)
            kernels_pred, out_chars, sub_img_nums = model(imgs, text_polys, is_train=False)

            loss_kernel = criterion_kernel(kernels_pred, kernel_labels)
            loss_kernel_item = loss_kernel.item()
            if (evel_step + 1) % 50 == 0:
                torch.cuda.empty_cache()
                is_print = False
            else:
                is_print = False
            loss_char, CR_correct_chars, AR_correct_chars, all_chars = ctc_loss(criterion_char, out_chars, label_tensors, text_lengths,
                                                                                sub_img_nums,
                                                                                char_set, is_print)

            a_CR_correct_chars += CR_correct_chars
            a_AR_correct_chars += AR_correct_chars
            a_all_chars += all_chars
            loss_char_item = loss_char.item()
            loss_all += 0.0 * loss_kernel_item + loss_char_item
            loss_char_all += loss_char_item
            loss_kernel_all += loss_kernel_item

            AR = a_AR_correct_chars / (a_all_chars + 1)
            CR = a_CR_correct_chars / (a_all_chars + 1)
            if evel_step % 10 == 0:
                pbar.display(
                    'eval epoch: {} '
                    'steps:{}/{} '
                    'loss_char_all:{:.6f} '
                    'loss_char:{:.4f} '
                    'loss_kernel_all:{:.6f} '
                    'loss_kernel:{:.4f} '
                    'AR:{:.4f} '

                    'CR:{:.4f} AR_all:{:.4f} '
                    'CR_all:{:.4f}\n'.
                        format(epoch,
                               evel_step,
                               evel_steps,
                               loss_char_all / (evel_step + 1),
                               loss_char_item,
                               loss_kernel_all / (evel_step + 1),
                               loss_kernel_item,

                               AR_correct_chars / all_chars,
                               CR_correct_chars / all_chars,
                               AR,
                               CR))

                pbar.update(10)
    pbar.close()
    AR = a_AR_correct_chars / (a_all_chars + 1)
    CR = a_CR_correct_chars / (a_all_chars + 1)
    global max_CR
    if is_save:
        if CR > max_CR:
            max_CR = CR
            torch.save(model.state_dict(), './output/with_tcn_big_hwdb_all_t/model_c_'
                                           'epoch_{}_'
                                           'loss_char_all_{:.4f}_'
                                           'loss_kernel_all_{:.4f}_'
                                           'AR_{:.6f}_'
                                           'CR_{:.6f}.pth'.format(epoch,
                                                                  loss_char_all / (evel_steps + 1),
                                                                  loss_kernel_all / (evel_steps + 1),
                                                                  AR,
                                                                  CR))
    log_writer.write('eval epoch:{} loss_kernel:{:.4f} loss_char:{:.4f} AR:{:.4f} CR:{:.4f}\n'.format(
        epoch,
        loss_kernel_all / (evel_steps + 1),
        loss_char_all / (evel_steps + 1),
        a_AR_correct_chars / a_all_chars,
        a_CR_correct_chars / a_all_chars))


def train(model, optimizer, train_dataset, criterion_kernel, criterion_char, epoch):
    train_dataset.epoch_count = epoch

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=AlignCollate(), batch_size=4, shuffle=True,
                                  num_workers=4, pin_memory=False)

    train_steps = len(train_dataloader)
    train_iter = iter(train_dataloader)
    model.train()
    # model.PAN_layer.eval()
    pbar = tqdm(total=train_steps)
    a_CR_correct_chars, a_AR_correct_chars, a_all_chars = 0, 0, 0
    loss_all = 0
    loss_char_all = 0
    loss_kernel_all = 0
    for train_step in range(train_steps):

        imgs, kernel_labels, text_polys, label_tensors, text_lengths, _ = next(train_iter)
        # torch.cuda.empty_cache()
        imgs = imgs.to(device)
        kernel_labels = kernel_labels.to(device)
        with autocast():
            kernels_pred, out_chars, sub_img_nums = model(imgs, text_polys, is_train=True)
            loss_kernel = criterion_kernel(kernels_pred, kernel_labels)

            loss_kernel_item = loss_kernel.item()

            if (train_step + 1) % 200 == 0:

                is_print = False
            else:
                is_print = False
            loss_char, CR_correct_chars, AR_correct_chars, all_chars = ctc_loss(criterion_char, out_chars, label_tensors, text_lengths,
                                                                                sub_img_nums,
                                                                                char_set, is_print)
            a_CR_correct_chars += CR_correct_chars
            a_AR_correct_chars += AR_correct_chars
            a_all_chars += all_chars
            loss_char_item = loss_char.item()
            loss_char_all += loss_char_item
            loss_kernel_all += loss_kernel_item

            if loss_kernel_item > 0.13:
                loss = 0.1 * loss_kernel + loss_char
            else:
                loss = loss_char

            loss_all += loss.item()
        optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        # pbar.set_description(str((loss_kernel.item(),loss_char.item(),loss_all/(train_step+1))))
        AR = a_AR_correct_chars / (a_all_chars + 1)
        CR = a_CR_correct_chars / (a_all_chars + 1)

        if (train_step + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (train_step + 1) % 10 == 0:
            pbar.display(
                'train epoch: {} '
                'steps:{}/{} '
                'loss_char_all:{:.6f} '
                'loss_char:{:.4f} '
                'loss_kernel_all:{:.6f} '
                'loss_kernel:{:.4f} '
                'AR:{:.4f} CR:{:.4f} '
                'AR_all:{:.4f} '
                'CR_all:{:.4f}\n'.
                    format(epoch,
                           train_step, train_steps,
                           loss_char_all / (train_step + 1),
                           loss_char_item,
                           loss_kernel_all / (train_step + 1),
                           loss_kernel_item,
                           AR_correct_chars / all_chars,
                           CR_correct_chars / all_chars,
                           AR,
                           CR))
            pbar.update(10)
    log_writer.write('train epoch:{} loss_kernel:{:.4f} loss_char:{:.4f} AR:{:.4f} CR:{:.4f}\n'.format(
        epoch,
        loss_kernel_all / (train_steps + 1),
        loss_char_all / (train_steps + 1),
        a_AR_correct_chars / a_all_chars,
        a_CR_correct_chars / a_all_chars))
    pbar.close()


if __name__ == '__main__':
    device = torch.device('cuda')
    max_CR = 0
    model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
    train_data = MyDataset(
        [
            # '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.0Test',
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.0Train',
            # '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.1Test',
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.1Train',
            # '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.2Test',
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.2Train',
            # './data/gen_for_icdar',
            # './data/gen_for_icdar1'
        ], char_dict,
        data_shape=1600, n=2, m=0.6,
        transform=transforms.ToTensor(), max_text_length=80)
    evel_data = MyDataset(
        [
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.0Test',
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.1Test',
            '/home/project/hwdb_dect_reco/data/hwdb2/HWDB2.2Test',

        ], char_dict,

        data_shape=1600, n=2, m=0.6,
        transform=transforms.ToTensor(), max_text_length=80, is_train=False)
    criterion_kernel = DICE_loss().to(device)
    criterion_char = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # pre_dict = torch.load(
    #     './output/with_tcn_big_icdar/model_new1_epoch_13_loss_char_all_0.3923_loss_kernel_all_0.1185_AR_0.911840_CR_0.920156.pth')
    #
    # # pre_dict.pop('DenseNet_layer.classifier.weight')
    # # pre_dict.pop('DenseNet_layer.classifier.bias')
    # model_dict = model.state_dict()
    # pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    #
    # model_dict.update(pre_dict)
    # model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load(
    #     r'./output/with_tcn_big_hwdb_all_t'
    #     r'/model_c_epoch_50_loss_char_all_0.0642_loss_kernel_all_0.1226_AR_0.987677_CR_0.990463.pth'))
    log_writer = open('./output/with_tcn_big_hwdb_all_t/log.txt', 'a', encoding='utf-8')

    # eval(model, evel_data, criterion_kernel, criterion_char, 0,is_save=False)

    for epoch in range(0, 50):
        optimizer = optim.Adam(
            model.parameters(), lr=0.0001 * 0.9 ** (epoch), betas=(0.5, 0.999))

        train(model, optimizer, train_data, criterion_kernel, criterion_char, epoch)

        eval(model, evel_data, criterion_kernel, criterion_char, epoch)
