import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)


class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.BCELoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets.unsqueeze(1).float())
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss      # CE+SCL


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    def forward(self, msg_emmbedding,code_emmbedding,cross_msg_code, manual_features=None, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        code_cls = code_emmbedding[:, 0, :]  # [bs, hidden_size]
        msg_cls = msg_emmbedding[:, 0, :]  # [bs, hidden_size]
        cross_cls = cross_msg_code[:, 0, :]  # [bs, hidden_size]
        concatenated_features = torch.cat((msg_cls, cross_cls,code_cls), dim=-1)# [bs, 3 * hidden_size]
        fused_features = self.fusion_mlp(concatenated_features)  # [bs, hidden_size]


        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)
        # x = torch.cat((x, y), dim=-1)
        # x = self.dropout(x)
        # x = self.out_proj_new(x)
        # return x

        combined = torch.cat((fused_features, y), dim=-1)
        combined_drop = self.dropout(combined)
        logits = self.out_proj_new(combined_drop)
        return combined, logits  # 返回融合后的特征和logits

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, patch_embedding, text_embedding):
        # batch_size, seq_length, dim = patch_embedding.size()
        # q = self.query(patch_embedding)
        # k = self.key(text_embedding)
        # v = self.value(text_embedding)

        batch_size, seq_length, dim = text_embedding.size()
        q = self.query(text_embedding)
        k = self.key(patch_embedding)
        v = self.value(patch_embedding)


        scores = torch.matmul(q, k.transpose(-2, -1)) / dim ** 0.5
        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, v)
        out = self.out(out)
        return out

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.cross_attention = CrossAttention(config.hidden_size)
        self.args = args
        self.con_loss = SupConLoss(alpha=args.alpha, temp=args.temp)

    def forward(self, inputs_ids,attn_masks, msg_input_ids,msg_input_mask,manual_features=None,
                labels=None, output_attentions=None):
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        msg_outputs = self.encoder(input_ids=msg_input_ids, attention_mask=msg_input_mask)
        msg_emmbedding = msg_outputs[0] #batch_size,msg_seq_len,hidden_size

        code_emmbedding = outputs[0] #batch_size,code_seq_len,hidden_size

        cross_msg_code = self.cross_attention(code_emmbedding,msg_emmbedding)#batch_size, code_seq_len, hidden_size

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None

        # logits = self.classifier(outputs[0], manual_features)
        features, logits = self.classifier(msg_emmbedding,code_emmbedding,cross_msg_code, manual_features)

        prob = torch.sigmoid(logits)
        if labels is not None:
            # loss_fct = BCELoss()
            # loss = loss_fct(prob, labels.unsqueeze(1).float())

            # 准备输出字典用于对比损失
            output_dict = {
                'predicts': prob,  # 使用DP任务的logits
                'cls_feats': features  # 使用融合后的特征
            }
            # 计算对比损失，只计算了dp任务的对比损失
            loss = self.con_loss(output_dict, labels)
            return loss, prob, last_layer_attn_weights
        else:
            return prob

