
import torch
import difflib
import numpy as np

def ctc_loss(criterion,prediction_chars, labels,label_lengths,sub_img_nums,char_set,is_print=False):

    cost_chars=[]
    a_CR_correct_chars, a_AR_correct_chars, a_all_chars=0,0,0
    sub_count=0

    for i in range(len(labels)):
        label = labels[i]
        label_length = label_lengths[i]
        prediction_char = prediction_chars[sub_count:sub_count+sub_img_nums[i]]
        sub_count += sub_img_nums[i]
        # print(prediction_char.shape)
        # print(label_length)

        input_length = torch.full(size=(len(prediction_char),), fill_value=prediction_char.shape[1], dtype=torch.long).to(
            prediction_char.device)
        prediction_char = prediction_char.log_softmax(-1)
        prediction_char = prediction_char.permute(1, 0, 2)

        torch.backends.cudnn.enabled = False
        # print(prediction_char.shape,label_length)

        cost_char = criterion(prediction_char, label, input_length, label_length)


        torch.backends.cudnn.enabled = True

        if torch.isnan(cost_char):
            print(prediction_char)
            print(label.shape)

            raise
        prediction_char = prediction_char.permute(1, 0, 2)
        if is_print:
            acc = ctc_acc(char_set, prediction_char, label, label_length, blank=0, p=True, bean_search=False)
            # print(acc)
        else:
            acc = ctc_acc(char_set, prediction_char, label, label_length, blank=0, p=False, bean_search=False)
        CR_correct_chars, AR_correct_chars, all_chars = acc[1]
        a_CR_correct_chars+=CR_correct_chars
        a_AR_correct_chars+=AR_correct_chars
        a_all_chars+=all_chars
        cost_chars.append(cost_char)
    cost_chars = torch.mean(torch.stack(cost_chars))
    return cost_chars,a_CR_correct_chars,a_AR_correct_chars,a_all_chars

def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def beam_decode_lite(y, beam_size=10):
    T, V = y.shape
    log_y = y
    top5 = partition_arg_topK(-y,beam_size,1)

    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(beam_size):  # for every state
                new_prefix = prefix + [top5[t][i]]
                new_score = score + log_y[t, top5[t][i]]

                new_beam.append((new_prefix, new_score))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)

        beam = new_beam[:beam_size]

    return beam


def ctc_acc(char_set,preds, labels, label_length,blank=0, p=True, bean_search=False):
    correct_words = 0
    CR_correct_chars = 0
    AR_correct_chars = 0
    all_chars = 0
    label_pred = []
    preds_softmax = torch.argmax(preds, -1)

    for i, label in enumerate(labels):
        sub_label_pred = []
        pred_softmax = preds_softmax[i]

        pred_s = remove_blank(pred_softmax.cpu().numpy())
        label = label[:label_length[i]]

        pred_str = ''
        label_str = ''

        for ci in pred_s:
            if ci!=blank and ci<len(char_set):
                pred_str += char_set[ci]
        for ci in label:
            if ci!=blank and ci<len(char_set):
                label_str += char_set[ci]
        # CR_correct_char = len(label_str)
        # AR_correct_char = len(label_str)
        sub_label_pred.append(label_str)
        if len(label_str) == len(pred_str):
            if label_str == pred_str:
                correct_words += 1

        CR_correct_char = max(len(label_str), len(pred_str))
        AR_correct_char = max(len(label_str), len(pred_str))
        # CR_correct_char = len(label_str)
        # AR_correct_char = len(label_str)

        for block in difflib.SequenceMatcher(None, label_str, pred_str).get_opcodes():
            label_m = block[2] - block[1]
            pred_m = block[4] - block[3]
            if block[0] == 'delete':
                CR_correct_char -= max(label_m,pred_m)
                AR_correct_char -= max(label_m,pred_m)

            elif block[0] == 'insert':
                AR_correct_char -= max(label_m,pred_m)

            elif block[0] == 'replace':

                CR_correct_char-=label_m




                AR_correct_char-=max(pred_m,label_m)

            # if block[0] in ['delete', 'replace', 'insert']:
            #     if block[0] == 'replace':
            #
            #         CR_correct_char -= max(block[2] - block[1], block[4] - block[3])
            #     else:
            #         CR_correct_char -= block[2] - block[1]
            #     if block[0] == 'insert':
            #         AR_correct_char -= block[4] - block[3]
            #     else:
            #         if block[0] == 'replace':
            #             AR_correct_char -= max(block[2] - block[1], block[4] - block[3])
            #         else:
            #             AR_correct_char -= block[2] - block[1]

        CR_correct_chars += CR_correct_char
        AR_correct_chars += AR_correct_char

        all_chars += max(len(label_str), len(pred_str))
        # all_chars+=len(label_str)
        if p:
            pred = preds[i]

            print(label_str, AR_correct_char / max(len(label_str),len(pred_str), 1),CR_correct_char / max(len(label_str),len(pred_str), 1))
            if bean_search:
                beans = beam_decode_lite(pred.cpu().detach().numpy(), beam_size=5)
                for bean, score in beans:
                    bean = remove_blank(bean)
                    bean_str = ''
                    for strii in bean:
                        try:
                            bean_str += char_set[strii]
                        except:
                            pass
                    sub_label_pred.append([bean_str, score])
                    print(bean_str, score)
                label_pred.append(sub_label_pred)
            else:
                print(pred_str)
            print()
    return correct_words / labels.shape[0], [CR_correct_chars, AR_correct_chars, all_chars], label_pred

# if __name__ == '__main__':
#     device = 'cuda'
#     criterion = torch.nn.CTCLoss(blank=1).to(device)
#     # Target are to be un-padded
#     T = 50  # Input sequence length
#     C = 20  # Number of classes (including blank)
#     N = 16  # Batch size
#     # Initialize random batch of input vectors, for *size = (T,N,C)
#     input = torch.randn(T, N, C).detach().requires_grad_()
#     input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
#     # Initialize random batch of targets (0 = blank, 1:C = classes)
#     target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
#     target = torch.randint(low=1, high=C, size=(N,T), dtype=torch.long)
#     ctc_loss(criterion,[input],[target],[input_lengths],[target_lengths])


import torch
import difflib
import numpy as np

def ctc_loss(criterion,prediction_chars, labels,label_lengths,sub_img_nums,char_set,is_print=False):

    cost_chars=[]
    a_CR_correct_chars, a_AR_correct_chars, a_all_chars=0,0,0
    sub_count=0

    for i in range(len(labels)):
        label = labels[i]
        label_length = label_lengths[i]
        prediction_char = prediction_chars[sub_count:sub_count+sub_img_nums[i]]
        sub_count += sub_img_nums[i]
        # print(prediction_char.shape)
        # print(label_length)

        input_length = torch.full(size=(len(prediction_char),), fill_value=prediction_char.shape[1], dtype=torch.long).to(
            prediction_char.device)
        prediction_char = prediction_char.log_softmax(-1)
        prediction_char = prediction_char.permute(1, 0, 2)

        torch.backends.cudnn.enabled = False
        # print(prediction_char.shape,label_length)

        cost_char = criterion(prediction_char, label, input_length, label_length)


        torch.backends.cudnn.enabled = True

        if torch.isnan(cost_char):
            print(prediction_char)
            print(label.shape)

            raise
        prediction_char = prediction_char.permute(1, 0, 2)
        if is_print:
            acc = ctc_acc(char_set, prediction_char, label, label_length, blank=0, p=True, bean_search=False)
            # print(acc)
        else:
            acc = ctc_acc(char_set, prediction_char, label, label_length, blank=0, p=False, bean_search=False)
        CR_correct_chars, AR_correct_chars, all_chars = acc[1]
        a_CR_correct_chars+=CR_correct_chars
        a_AR_correct_chars+=AR_correct_chars
        a_all_chars+=all_chars
        cost_chars.append(cost_char)
    cost_chars = torch.mean(torch.stack(cost_chars))
    return cost_chars,a_CR_correct_chars,a_AR_correct_chars,a_all_chars

def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def beam_decode_lite(y, beam_size=10):
    T, V = y.shape
    log_y = y
    top5 = partition_arg_topK(-y,beam_size,1)

    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(beam_size):  # for every state
                new_prefix = prefix + [top5[t][i]]
                new_score = score + log_y[t, top5[t][i]]

                new_beam.append((new_prefix, new_score))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)

        beam = new_beam[:beam_size]

    return beam


def ctc_acc(char_set,preds, labels, label_length,blank=0, p=True, bean_search=False):
    correct_words = 0
    CR_correct_chars = 0
    AR_correct_chars = 0
    all_chars = 0
    label_pred = []
    preds_softmax = torch.argmax(preds, -1)

    for i, label in enumerate(labels):
        sub_label_pred = []
        pred_softmax = preds_softmax[i]

        pred_s = remove_blank(pred_softmax.cpu().numpy())
        label = label[:label_length[i]]

        pred_str = ''
        label_str = ''

        for ci in pred_s:
            if ci!=blank and ci<len(char_set):
                pred_str += char_set[ci]
        for ci in label:
            if ci!=blank and ci<len(char_set):
                label_str += char_set[ci]
        # CR_correct_char = len(label_str)
        # AR_correct_char = len(label_str)
        sub_label_pred.append(label_str)
        if len(label_str) == len(pred_str):
            if label_str == pred_str:
                correct_words += 1

        CR_correct_char = max(len(label_str), len(pred_str))
        AR_correct_char = max(len(label_str), len(pred_str))
        # CR_correct_char = len(label_str)
        # AR_correct_char = len(label_str)

        for block in difflib.SequenceMatcher(None, label_str, pred_str).get_opcodes():
            label_m = block[2] - block[1]
            pred_m = block[4] - block[3]
            if block[0] == 'delete':
                CR_correct_char -= max(label_m,pred_m)
                AR_correct_char -= max(label_m,pred_m)

            elif block[0] == 'insert':
                AR_correct_char -= max(label_m,pred_m)

            elif block[0] == 'replace':

                CR_correct_char-=label_m




                AR_correct_char-=max(pred_m,label_m)

            # if block[0] in ['delete', 'replace', 'insert']:
            #     if block[0] == 'replace':
            #
            #         CR_correct_char -= max(block[2] - block[1], block[4] - block[3])
            #     else:
            #         CR_correct_char -= block[2] - block[1]
            #     if block[0] == 'insert':
            #         AR_correct_char -= block[4] - block[3]
            #     else:
            #         if block[0] == 'replace':
            #             AR_correct_char -= max(block[2] - block[1], block[4] - block[3])
            #         else:
            #             AR_correct_char -= block[2] - block[1]

        CR_correct_chars += CR_correct_char
        AR_correct_chars += AR_correct_char

        all_chars += max(len(label_str), len(pred_str))
        # all_chars+=len(label_str)
        if p:
            pred = preds[i]

            print(label_str, AR_correct_char / max(len(label_str),len(pred_str), 1),CR_correct_char / max(len(label_str),len(pred_str), 1))
            if bean_search:
                beans = beam_decode_lite(pred.cpu().detach().numpy(), beam_size=5)
                for bean, score in beans:
                    bean = remove_blank(bean)
                    bean_str = ''
                    for strii in bean:
                        try:
                            bean_str += char_set[strii]
                        except:
                            pass
                    sub_label_pred.append([bean_str, score])
                    print(bean_str, score)
                label_pred.append(sub_label_pred)
            else:
                print(pred_str)
            print()
    return correct_words / labels.shape[0], [CR_correct_chars, AR_correct_chars, all_chars], label_pred

# if __name__ == '__main__':
#     device = 'cuda'
#     criterion = torch.nn.CTCLoss(blank=1).to(device)
#     # Target are to be un-padded
#     T = 50  # Input sequence length
#     C = 20  # Number of classes (including blank)
#     N = 16  # Batch size
#     # Initialize random batch of input vectors, for *size = (T,N,C)
#     input = torch.randn(T, N, C).detach().requires_grad_()
#     input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
#     # Initialize random batch of targets (0 = blank, 1:C = classes)
#     target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
#     target = torch.randint(low=1, high=C, size=(N,T), dtype=torch.long)
#     ctc_loss(criterion,[input],[target],[input_lengths],[target_lengths])

