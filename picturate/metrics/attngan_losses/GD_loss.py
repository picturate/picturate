from picturate.imports import *
from picturate.metrics.attngan_losses import *


def attngan_discriminator_loss(
    netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels
):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(
        real_features[: (batch_size - 1)], conditions[1:batch_size]
    )
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = (real_errD + cond_real_errD) / 2.0 + (
            fake_errD + cond_fake_errD + cond_wrong_errD
        ) / 3.0
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.0
    return errD


def attngan_generator_loss(
    netsD,
    image_encoder,
    fake_imgs,
    real_labels,
    words_embs,
    sent_emb,
    match_labels,
    cap_lens,
    class_ids,
):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ""
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.item()
        logs += "g_loss%d: %.2f " % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(
                region_features,
                words_embs,
                match_labels,
                cap_lens,
                class_ids,
                batch_size,
            )
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.item()

            s_loss0, s_loss1 = sent_loss(
                cnn_code, sent_emb, match_labels, class_ids, batch_size
            )
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.item()

            errG_total += w_loss + s_loss
            logs += "w_loss: %.2f s_loss: %.2f " % (w_loss.item(), s_loss.item())
    return errG_total, logs


def cycle_generator_loss(
    netsD,
    image_encoder,
    fake_imgs,
    real_labels,
    captions,
    words_embs,
    sent_emb,
    match_labels,
    cap_lens,
    class_ids,
):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ""
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.item()
        logs += "g_loss%d: %.2f " % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code, word_logits = image_encoder(
                fake_imgs[i], captions
            )
            w_loss0, w_loss1, _ = words_loss(
                region_features,
                words_embs,
                match_labels,
                cap_lens,
                class_ids,
                batch_size,
            )
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.item()

            s_loss0, s_loss1 = sent_loss(
                cnn_code, sent_emb, match_labels, class_ids, batch_size
            )
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.item()

            t_loss = image_to_text_loss(word_logits, captions) * cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w_loss + s_loss + t_loss
            logs += "w_loss: %.2f s_loss: %.2f t_loss: %.2f" % (
                w_loss.item(),
                s_loss.item(),
                t_loss.item(),
            )
    return errG_total, logs
