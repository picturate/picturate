from picturate.imports import *
from picturate.config import *
from picturate.nets.cycle_attngan.attention import GlobalAttentionGeneral as ATT_NET
from picturate.nets.cycle_attngan.encoders import GLU, upBlock, ResBlock, conv3x3

"""
Code based on the repository by suetAndTie
https://github.com/suetAndTie/cycle-image-gan
"""

__all__ = ["G_NET"]

# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, cfg):
        super(CA_NET, self).__init__()
        self.cfg = cfg
        self.t_dim = self.cfg.TEXT.EMBEDDING_DIM
        self.c_dim = self.cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, : self.c_dim]
        logvar = x[:, self.c_dim :]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cfg.CUDA:

            eps = torch.cuda.FloatTensor(std.size()).normal_()
            mu = mu.cuda()
            std = std.cuda()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf, cfg):
        super(INIT_STAGE_G, self).__init__()
        self.cfg = cfg
        self.gf_dim = ngf
        self.in_dim = self.cfg.GAN.Z_DIM + ncf  # self.cfg.TEXT.EMBEDDING_DIM

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU(),
        )

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x self.cfg.GAN.Z_DIM
        :param c_code: batch x self.cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, cfg):
        super(NEXT_STAGE_G, self).__init__()
        self.cfg = cfg
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = self.cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf, cfg):
        super(GET_IMAGE_G, self).__init__()
        self.cfg = cfg
        self.gf_dim = ngf
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self, cfg):
        super(G_NET, self).__init__()
        self.cfg = cfg
        ngf = self.cfg.GAN.GF_DIM
        nef = self.cfg.TEXT.EMBEDDING_DIM
        ncf = self.cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET(self.cfg)

        if self.cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf, self.cfg)
            self.img_net1 = GET_IMAGE_G(ngf, self.cfg)
        # gf x 64 x 64
        if self.cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, self.cfg)
            self.img_net2 = GET_IMAGE_G(ngf, self.cfg)
        if self.cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, self.cfg)
            self.img_net3 = GET_IMAGE_G(ngf, self.cfg)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x self.cfg.GAN.Z_DIM
            :param sent_emb: batch x self.cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if self.cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if self.cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


class G_DCGAN(nn.Module):
    def __init__(self, cfg):
        super(G_DCGAN, self).__init__()
        self.cfg = cfg
        ngf = self.cfg.GAN.GF_DIM
        nef = self.cfg.TEXT.EMBEDDING_DIM
        ncf = self.cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET(self.cfg)

        # 16gf x 64 x 64 --> gf x 64 x 64 --> 3 x 64 x 64
        if self.cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf, self.cfg)
        # gf x 64 x 64
        if self.cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, self.cfg)
        if self.cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, self.cfg)
        self.img_net = GET_IMAGE_G(ngf, self.cfg)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x self.cfg.GAN.Z_DIM
            :param sent_emb: batch x self.cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if self.cfg.TREE.BRANCH_NUM > 0:
            h_code = self.h_net1(z_code, c_code)
        if self.cfg.TREE.BRANCH_NUM > 1:
            h_code, att1 = self.h_net2(h_code, c_code, word_embs, mask)
            if att1 is not None:
                att_maps.append(att1)
        if self.cfg.TREE.BRANCH_NUM > 2:
            h_code, att2 = self.h_net3(h_code, c_code, word_embs, mask)
            if att2 is not None:
                att_maps.append(att2)

        fake_imgs = self.img_net(h_code)
        return [fake_imgs], att_maps, mu, logvar


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid()
        )

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, cfg, b_jcu=True):
        super(D_NET64, self).__init__()
        self.cfg = cfg
        ndf = self.cfg.GAN.DF_DIM
        nef = self.cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code_s16(x_var)  # 4 x 4 x 8df
        return x_code4


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, cfg, b_jcu=True):
        super(D_NET128, self).__init__()
        self.cfg = cfg
        ndf = self.cfg.GAN.DF_DIM
        nef = self.cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)  # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)  # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, cfg, b_jcu=True):
        super(D_NET256, self).__init__()
        self.cfg = cfg
        ndf = self.cfg.GAN.DF_DIM
        nef = self.cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4
