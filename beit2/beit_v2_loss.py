import torch

class BeitV2Loss(torch.nn.CrossEntropyLoss): 
    """MSE loss is calculated on the masked patches only
       supports the following modes:
           masked - MSE on masked areas - the 'proper' loss
           full - MSE on the full image
           visible - MSE only on the visible patches - sanity check
    """
    def __init__(
            self,
            vis_weight: float=0.,
            calc_unweighted_loss: bool=False,
            vis_mask_key: str="vis_mask",
            invis_mask_key: str="invis_mask",
            **super_kwargs
        ):
        self._ignore_index = 100000
        super(BeitV2Loss, self).__init__(ignore_index=self._ignore_index, **super_kwargs)
        assert 0 <= vis_weight <= 1, "vis_weight is the relative weight between visible patches and invisible ones. therefore it can only be in [0,1]" 
        self.vis_weight = vis_weight 
        self.invis_weight = 1-vis_weight 
        self.calc_unweighted_loss = calc_unweighted_loss 
        self.vis_mask_key = vis_mask_key 
        self.invis_mask_key = invis_mask_key 

    def __call__(self, pred, batch): 
        # visible loss: 
        if self.vis_weight > 0 or self.calc_unweighted_loss: 
            vis_mask = batch[self.vis_mask_key].flatten(1) 
            vis_loss = self.call_on_mask(pred, vis_mask) 
        else: 
            vis_loss = torch.tensor(0., device=batch[self.vis_mask_key].device) 

        # invisible loss: 
        if self.invis_weight > 0 or self.calc_unweighted_loss: 
            invis_mask = batch[self.invis_mask_key].flatten(1) 
            invis_loss = self.call_on_mask(pred, invis_mask) 
        else: 
            invis_loss = torch.tensor(0., device=batch[self.invis_mask_key].device) 

        weighted_loss = self.vis_weight * vis_loss + self.invis_weight * invis_loss 

        return {
            "vis_loss": vis_loss,
            "invis_loss": invis_loss,
            "loss": weighted_loss
        }

    def mask_labels(self, labels, mask): 
        input_dim = labels.shape 
        addition_mask = ~mask * self._ignore_index 
        return ((labels * mask) + addition_mask).reshape((input_dim[0]*input_dim[1], )) 
        
    def mask_pred(self, pred, mask): 
        input_dim = pred.shape 
        mask = mask.unsqueeze(2).repeat((1, 1, input_dim[-1])) 
        addition_mask = ~mask * self._ignore_index 
        return ((pred * mask) + addition_mask).reshape((input_dim[0] * input_dim[1], input_dim[2])) 

    def call_on_mask(self, pred, mask): 
        visible_tokenizer_labels = self.mask_labels(pred['tokenizer_labels'], mask) 
        visible_model_outputs = self.mask_pred(pred['beit_model_output'], mask) 
        vis_loss = super(BeitV2Loss, self).__call__(input=visible_model_outputs, target=visible_tokenizer_labels) 
        return vis_loss
