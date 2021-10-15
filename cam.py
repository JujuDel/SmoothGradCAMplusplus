import torch
import torch.nn.functional as F

from statistics import mode, mean


class SaveValues():
    def __init__(self, m, verbose=False):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        if verbose:
            print('Registering hooks on:\n', m)
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        if self.activations is None:
            self.activations = [output.detach().cpu().clone()]
        else:
            self.activations += [output.detach().cpu().clone()]

    def hook_fn_grad(self, module, grad_input, grad_output):
        if self.gradients is None:
            self.gradients = [grad_output[0].detach().cpu().clone()]
        else:
            self.gradients += [grad_output[0].detach().cpu().clone()]

    def remove(self):
        if hasattr(self, "forward_hook"):
            self.forward_hook.remove()
        if hasattr(self, "backward_hook"):
            self.backward_hook.remove()

    def clean(self):
        if self.activations is not None and isinstance(self.activations, list):
            del self.activations
            self.activations = None
        if self.gradients is not None and isinstance(self.gradients, list):
            del self.gradients
            self.gradients = None
        torch.cuda.empty_cache()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer, verbose=False):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        if target_layer is not None:
            self.values = SaveValues(self.target_layer, verbose)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, idx].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class OD_Loss(object):
    def __init__(self, special):
        self.special = special

    def __call__(self, model, img_tensor, meta, clsToKeep=-1, lossToKeep=None):
        if self.special == 'nanodet':
            preds = model.model.forward(img_tensor, extractAll=True)
            loss, loss_states = model.model.head.loss(preds["head"], meta, clsToKeep=clsToKeep)

            # Default loss is quality_focal_loss + GIoULoss_loss + distribution_focal_loss
            # loss_qfl + loss_bbox + loss_dfl
            if lossToKeep is not None and lossToKeep.lower() != 'all':
                loss = loss_states[lossToKeep]

            return loss

        elif self.special == 'mmdetection':
            # Backbone + FPN
            x = model.extract_feat(img_tensor)
            # Head
            outs = model.bbox_head(x)

            del img_tensor
            torch.cuda.empty_cache()

            # Head outs -> losses
            loss_inputs = outs + (meta["gt_bboxes"], meta["gt_labels"], meta["img_metas"])
            losses = model.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=None, clsToKeep=clsToKeep)
            loss, log_vars = model._parse_losses(losses)

            # Default loss is quality_focal_loss + GIoULoss_loss + distribution_focal_loss
            # loss_qfl + loss_bbox + loss_dfl
            if lossToKeep is not None and lossToKeep.lower() != 'all':
                try:
                    loss = losses[lossToKeep]
                except KeyError:
                    print(f'keyError on lossToKeep: {lossToKeep}')
                    print(' Possible choices are [' + ', '.join(['all'] + list(losses.keys())) + ']')
                    raise KeyError
            else:
                pass
            return loss

        else:
            raise NotImplementedError

class SmoothGradCAMpp(CAM):
    """ Smooth Grad CAM plus plus """

    def __init__(self, model, n_samples=25, stdev_spread=0.15, special=None, verbose=False):
        super().__init__(model, None, verbose)
        """
        Args:
            model: a base model
            n_sample: the number of samples
            stdev_spread: standard deviationß
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread
        self.special = special

    def forward_special(self, meta, shared_depth, clsToKeep=-1, lossToKeep=None, device=torch.device('cuda:0')):
        loss_calculator = OD_Loss(self.special)

        meta["img"] = meta["img"].to(device=device, non_blocking=True)

        stdev = self.stdev_spread / (meta["img"].max() - meta["img"].min())
        std_tensor = torch.ones_like(meta["img"]) * stdev

        for i in range(self.n_samples):
            self.model.zero_grad()
            self.values.clean()

            x_with_noise = torch.normal(mean=meta["img"], std=std_tensor)
            x_with_noise.requires_grad_()

            loss = loss_calculator(self.model, x_with_noise, meta, clsToKeep, lossToKeep)

            loss.backward(retain_graph=True)
            loss = loss.cpu()

            activations = self.values.activations[shared_depth]
            gradients = self.values.gradients[-1 - shared_depth]
            n, c, _, _ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            # calculate weights
            relu_grad = F.relu(loss.exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

            # shape => (1, 1, H', W')
            cam = (weights * activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

            if i == 0:
                total_cams = cam.clone()
            else:
                total_cams += cam

            self.values.clean()

        total_cams /= self.n_samples

        return total_cams.data

    def forward(self, x, shared_depth, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        indices = []
        probs = []

        for i in range(self.n_samples):
            self.model.zero_grad()
            self.values.clean()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            prob = F.softmax(score, dim=1)

            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                probs.append(prob.item())

            indices.append(idx)

            score[0, idx].backward(retain_graph=True)

            activations = self.values.activations[shared_depth]
            gradients = self.values.gradients[-1 - shared_depth]
            n, c, _, _ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            relu_grad = F.relu(score[0, idx].exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

            # shape => (1, 1, H', W')
            cam = (weights * activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

            if i == 0:
                total_cams = cam.clone()
            else:
                total_cams += cam

        total_cams /= self.n_samples
        idx = mode(indices)
        prob = mean(probs)

        print("predicted class ids {}\t probability {}".format(idx, prob))

        return total_cams.data, idx

    def __call__(self, x, target_layer, verbose=False, shared_depth=0, **kwargs):
        self.values = SaveValues(target_layer, verbose)
        try:
            if self.special is None:
                return self.forward(x, shared_depth)
            else:
                return self.forward_special(x, shared_depth, **kwargs)
        except NotImplementedError:
            self.values.remove()
            raise NotImplementedError
        self.values.remove()
