import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class CLIPLoss(nn.Module):
    """CLIP loss with reconstruction loss"""
    def __init__(self, 
                w1=1.0, 
                w2=1.0, 
                w3=1.0, 
                disable_clip=False,
                disable_profile=False,
                disable_text=False,
                mask_rate_max_threshold_clip=1.0, 
                mask_rate_min_threshold_recon=0.0,
                distributed=True, 
                learnable_logit_scale=False,
                soft_contrastive_loss_text=False,
                soft_contrastive_loss_profile=False):
        """
        Args:
            w1: weight for the image reconstruction loss
            w2: weight for the image-text contrastive loss
            w3: weight for the image-profile contrastive loss
        """
        super(CLIPLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.mask_rate_max_threshold_clip = mask_rate_max_threshold_clip
        self.mask_rate_min_threshold_recon = mask_rate_min_threshold_recon
        self.disable_clip = disable_clip
        self.disable_profile = disable_profile
        self.disable_text = disable_text
        self.distributed = distributed
        self.learnable_logit_scale = learnable_logit_scale
        self.default_logit_scale = 1 / 0.07
        self.soft_contrastive_loss_text = soft_contrastive_loss_text
        self.soft_contrastive_loss_profile = soft_contrastive_loss_profile

    def _compute_dsc_similarity(self, feat1, feat2):
        # feat1 / feat2: [num_samples, num_features]
        # dsc similarity = 2 * sum_i(v1[i] * v2[i]) / (sum_i(v1[i]) + sum_i(v2[i])) for binary vectors
        
        intersections = torch.matmul(feat1, feat2.T)
        sum1 = feat1.sum(dim=-1)
        sum2 = feat2.sum(dim=-1)
        # denominaator: sum_i(v1[i]) + sum_i(v2[j]) for all pairs
        denominator = sum1.unsqueeze(1) + sum2.unsqueeze(0)
        # Compute DSC similarity: 2 * intersections / denominator
        # Add small epsilon to avoid division by zero
        score = 2.0 * intersections / (denominator + 1e-8)
        
        return score

    def _compute_cosine_similarity_score(self, feat1, feat2):
        # feat1 / feat2: [num_samples, num_features]
        feat1 = F.normalize(feat1, dim=-1, p=2)
        feat2 = F.normalize(feat2, dim=-1, p=2)
        # Calculate pairwise cosine similarity between pretrained features
        score = torch.matmul(feat1, feat2.T)
        # Scale from [-1, 1] to [0, 1]
        score = (score + 1.0) / 2.0
        return score


    def _compute_contrastive_loss(self, feat1, feat2, logit_scale, valid_mask=None, feat2_pretrained=None, soft_label_type='none', debug=False):
        """
        Compute contrastive loss between two feature embeddings.
        
        Args:
            feat1: First feature embeddings (e.g., image features)
            feat2: Second feature embeddings (e.g., text or profile features)
            logit_scale: Scale factor for logits
            valid_mask: Optional mask indicating valid samples (default: all valid)
            feat2_pretrained: Pretrained feature embeddings (e.g., text global feature)
            similarity_type: Type of similarity to use ('cosine' or 'dsc'), 'none' for hard labels
            
        Returns:
            tuple: (contrastive_loss, num_valid_samples)
                - contrastive_loss: Loss value (0.0 if no valid samples)
                - num_valid_samples: Number of valid samples in this batch
        """

        if valid_mask is None:
            valid_mask = torch.ones(feat1.size(0), device=feat1.device).bool()
        else:
            valid_mask = valid_mask.bool()

        feat1 = F.normalize(feat1, dim=-1, p=2)
        feat2 = F.normalize(feat2, dim=-1, p=2)

        if self.distributed:
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            if feat2_pretrained is not None:
                all_feat2_pretrained = torch.cat(torch.distributed.nn.all_gather(feat2_pretrained), dim=0)
            else:
                all_feat2_pretrained = None
            all_valid_masks = torch.cat(torch.distributed.nn.all_gather(valid_mask), dim=0)
            rank = torch.distributed.get_rank()
            local_size = feat1.size(0)
            labels = torch.arange(local_size, device=feat1.device) + rank * local_size
        else:
            all_feat1 = feat1
            all_feat2 = feat2
            all_feat2_pretrained = feat2_pretrained
            all_valid_masks = valid_mask
            rank = 0
            local_size = feat1.size(0)
            labels = torch.arange(local_size, device=feat1.device)

        if debug:
            print(f"[CLIPLoss] all_valid_masks shape: {tuple(all_valid_masks.shape)}")


        # compute the logits for contrasting
        logits_1_per_2 = logit_scale * torch.matmul(feat1, all_feat2.T)
        logits_2_per_1 = logit_scale * torch.matmul(feat2, all_feat1.T)


        # Mask invalid samples in logits: set to very negative value so they won't be selected
        # all_valid_masks: (total_samples,) - True for valid samples
        # For logits_1_per_2: (local_size, total_samples) - mask columns (negatives)
        # For logits_2_per_1: (local_size, total_samples) - mask columns (negatives)
        invalid_mask = ~all_valid_masks  # (total_samples,)
        neg_large = torch.finfo(logits_1_per_2.dtype).min
        logits_1_per_2 = logits_1_per_2.masked_fill(invalid_mask.unsqueeze(0), neg_large)
        logits_2_per_1 = logits_2_per_1.masked_fill(invalid_mask.unsqueeze(0), neg_large)

        # Compute per-sample losses (reduction='none')
        if soft_label_type != 'none':
            assert feat2_pretrained is not None, "Pretrained features are required for soft labels"
            if soft_label_type == 'cosine':
                soft_label_1_per_2 = self._compute_cosine_similarity_score(feat2_pretrained, all_feat2_pretrained)
            elif soft_label_type == 'dsc':
                soft_label_1_per_2 = self._compute_dsc_similarity(feat2_pretrained, all_feat2_pretrained)
            else:
                raise ValueError(f"Invalid similarity type: {soft_label_type}")
            # Mask invalid samples to 0
            soft_label_1_per_2 = soft_label_1_per_2.masked_fill(invalid_mask.unsqueeze(0), 0.0)
            # Normalize to get probability distribution (handle zero sum case)
            soft_label_sum = soft_label_1_per_2.sum(dim=-1, keepdim=True)
            soft_label_1_per_2 = soft_label_1_per_2 / (soft_label_sum + 1e-8)
            # Use KL divergence for soft labels: KL(soft_label || softmax(logits))
            log_probs_1_per_2 = F.log_softmax(logits_1_per_2, dim=-1)
            loss_1_per_sample = F.kl_div(log_probs_1_per_2, soft_label_1_per_2, reduction='none').sum(dim=-1)
        else:
            # Use cross entropy with hard labels
            loss_1_per_sample = F.cross_entropy(logits_1_per_2, labels, reduction='none')
        
        loss_2_per_sample = F.cross_entropy(logits_2_per_1, labels, reduction='none')

        # Apply valid_mask to only compute loss for valid local samples
        loss_1 = (loss_1_per_sample * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
        loss_2 = (loss_2_per_sample * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
        
        contrastive_loss = (loss_1 + loss_2) / 2

        num_valid = valid_mask.float().sum()
        
        return contrastive_loss, num_valid

    def forward(self, outputs, debug=False):
        # Use separate image embeddings for text and profile alignments
        v_feats_text = outputs['image_emb_text']
        v_feats_profile = outputs['image_emb_profile']
        t_feats = outputs['text_emb']
        p_feats = outputs['profile_emb']
        img_recon_loss = outputs['img_recon_loss']
        logit_scale = outputs['logit_scale']
        logit_scale_profile = outputs['logit_scale_profile']
        report_valid = outputs['report_valid']
        text_global_feature_pretrained = outputs['text_global_feature_pretrained']
        one_hot_ehr = outputs['one_hot_ehr']
        mask_rate = outputs['mask_rate'] # (batch_size,)

        if not self.learnable_logit_scale:
            logit_scale = torch.tensor(self.default_logit_scale, device=logit_scale.device, dtype=logit_scale.dtype)
            logit_scale_profile = torch.tensor(self.default_logit_scale, device=logit_scale_profile.device, dtype=logit_scale_profile.dtype)
        
        recon_valid = mask_rate >= self.mask_rate_min_threshold_recon
        img_recon_valid_count = torch.tensor(recon_valid.float().sum(), device=img_recon_loss.device, dtype=torch.float32)

        clip_valid = mask_rate <= self.mask_rate_max_threshold_clip # (batch_size,)
        # print('DEBUG clip_valid:', clip_valid.shape, clip_valid.sum())
        # image-text contrastive loss
        if self.disable_clip or self.disable_text:
            # Set img_txt_loss to zero when CLIP is disabled
            img_txt_loss = torch.tensor(0.0, device=img_recon_loss.device, dtype=img_recon_loss.dtype)
            img_txt_valid_count = torch.tensor(0.0, device=img_recon_loss.device, dtype=torch.float32)
        else:
            # Apply mask_rate filtering: if mask_rate > threshold, set report_valid to all zeros
            report_valid = report_valid * clip_valid
            # print('DEBUG report_valid:', report_valid.shape, report_valid.sum())
            soft_label_type = 'cosine' if self.soft_contrastive_loss_text else 'none'
            img_txt_loss, img_txt_valid_count = self._compute_contrastive_loss(v_feats_text, t_feats, logit_scale,
                valid_mask=report_valid, feat2_pretrained=text_global_feature_pretrained, soft_label_type=soft_label_type, debug=debug)
        
        if self.disable_clip or self.disable_profile:
            # Set img_profile_loss to zero when profile is disabled
            img_profile_loss = torch.tensor(0.0, device=img_recon_loss.device, dtype=img_recon_loss.dtype)
            img_profile_valid_count = torch.tensor(0.0, device=img_recon_loss.device, dtype=torch.float32)
        else:
            # Apply mask_rate filtering: if mask_rate > threshold, set profile_valid to all zeros
            profile_valid_mask = one_hot_ehr.sum(dim=1) > 0  # [B] - True if any profile field is valid
            profile_valid_mask = profile_valid_mask * clip_valid
            # print('DEBUG profile_valid_mask:', profile_valid_mask.shape, profile_valid_mask.sum())
            soft_label_type = 'dsc' if self.soft_contrastive_loss_profile else 'none'
            img_profile_loss, img_profile_valid_count = self._compute_contrastive_loss(v_feats_profile, p_feats, logit_scale_profile,
                valid_mask=profile_valid_mask, feat2_pretrained=one_hot_ehr, soft_label_type=soft_label_type, debug=debug)

        # print(f"Rank {rank}: img_recon_loss: {img_recon_loss}, img_txt_loss: {img_txt_loss}, img_profile_loss: {img_profile_loss}")

        # total loss
        loss = self.w1 * img_recon_loss + self.w2 * img_txt_loss + self.w3 * img_profile_loss

        
        return {'loss': loss,
                'img_recon_loss': img_recon_loss,
                'img_txt_loss': img_txt_loss,
                'img_profile_loss': img_profile_loss,
                'img_recon_valid_count': img_recon_valid_count,
                'img_txt_valid_count': img_txt_valid_count,
                'img_profile_valid_count': img_profile_valid_count}