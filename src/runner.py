from http.client import NotConnected
from typing import Dict
from dataclasses import dataclass

import torch
import math
import numpy as np
import copy
#from functorch import * 
from torch.autograd.functional import *

import jiant.tasks.evaluate as evaluate
import jiant.utils.torch_utils as torch_utils
#from jiant.proj.main.components.container_setup import JiantTaskContainer
from container_setup import JiantTaskContainer # maple
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
from jiant.shared.constants import PHASE
from jiant.shared.runner import (
    #complex_backpropagate,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from jiant.utils.display import maybe_tqdm
from jiant.utils.python.datastructures import InfiniteYield, ExtendedDataClassMixin

def complex_backpropagate(
    loss, optimizer, model, fp16, n_gpu, gradient_accumulation_steps, max_grad_norm, retain_graph = False
):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return loss


@dataclass
class RunnerParameters(ExtendedDataClassMixin):
    local_rank: int
    n_gpu: int
    fp16: bool
    max_grad_norm: float


@dataclass
class TrainState(ExtendedDataClassMixin):
    global_steps: int
    task_steps: Dict[str, int]

    @classmethod
    def from_task_name_list(cls, task_name_list):
        return cls(global_steps=0, task_steps={task_name: 0 for task_name in task_name_list})

    def step(self, task_name):
        self.task_steps[task_name] += 1
        self.global_steps += 1

# Maple for hvp
# Following are utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def get_parms(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    parms = []
    names = []
    for name, p in list(mod.named_parameters()):
        parms.append(copy.deepcopy(p))
        del_attr(mod, name.split("."))
        names.append(name)
    return parms, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class JiantRunner:
    def __init__(
        self,
        jiant_task_container: JiantTaskContainer,
        jiant_model: JiantModel,
        optimizer_scheduler,
        device,
        rparams: RunnerParameters,
        log_writer,
    ):
        self.jiant_task_container = jiant_task_container
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.device = device
        self.rparams = rparams
        self.log_writer = log_writer

        self.model = self.jiant_model


    def run_train(self):
        for _ in self.run_train_context():
            pass

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        train_state = TrainState.from_task_name_list(
            self.jiant_task_container.task_run_config.train_task_list
        )
        pbar = maybe_tqdm(
            range(self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            verbose=verbose,
        )
        for _ in pbar:
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                pbar = pbar
            )
            yield train_state

    def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        start_position = train_state.global_steps
        pbar = maybe_tqdm(
            range(start_position, self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            initial=start_position,
            total=self.jiant_task_container.global_train_config.max_steps,
            verbose=verbose,
        )
        for _ in pbar:
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                pbar = pbar
            )
            yield train_state

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState, pbar):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        """if 'sctmask' in self.user_mode:
            retain_graph = True  
        else:
            retain_graph = False """
        if 'sctmask' in self.user_mode:
            #import numpy as np
            parms = dict(self.jiant_model.named_parameters())
            for name in parms:
                if not name.startswith('encoder') or '__' in name:
                    continue
                xname = "sctmask__" + name.replace(".", "__")
                idx = self.jiant_model.encoder.idx_dict[xname]
                x = getattr(self.jiant_model.encoder, xname)
                parms[name].detach_()
                parms[name].flatten()[idx] = x
        elif 'prompt' in self.user_mode:
            self.jiant_model.encoder.embeddings.word_embeddings.weight.detach_()
            self.jiant_model.encoder.embeddings.word_embeddings.weight[:] = torch.cat([self.jiant_model.encoder.embeddings.word_embeddings.weight_ori, self.jiant_model.encoder.embeddings.prompt_weight])
        elif 'diffprun' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "dp_burnin_step"):
                self.dp_burnin_step = 500 if 'burnin' not in self.user_mode else int(self.user_mode['burnin'])
                self.dp_mask = {}
                self.dp_step = 0
                for p in parms:
                    if not p.startswith('encoder.w'):
                        continue
                    pn = p.replace("encoder.w__", "").replace("__", ".")
                    self.dp_mask[pn] = torch.zeros_like(parms[p])
            self.dp_step += 1
            if self.dp_step < self.dp_burnin_step:
                l0s = []
                for p in parms:
                    if not p.startswith('encoder.w'):
                        continue
                    p = p.replace("encoder.w__", "").replace("__", ".")
                    alpha = getattr(self.jiant_model.encoder, "ber__" + p.replace(".", "__"))
                    w = getattr(self.jiant_model.encoder, "w__" + p.replace(".", "__"))
                    u = 1e-6 + torch.rand_like(w) * (1- 2e-6)
                    st = torch.sigmoid(torch.log(u) - torch.log(1-u) + alpha)
                    clamp = float(self.user_mode['clamp']) if 'clamp' in self.user_mode else 3.
                    l, r = -clamp, clamp
                    stb = st * (r-l) + l
                    z = stb.clamp_min(0).clamp_max(1)
                    nw = self.jiant_model.ori_pars[p] + z * w
                    node = self.jiant_model
                    pnames = p.split(".")
                    for pname in pnames[:-1]:
                        node = getattr(node, pname)
                    delattr(node, pnames[-1]) 
                    setattr(node, pnames[-1], nw)
                    l0s.append(torch.sigmoid(alpha - math.log(-l/r)).flatten())
                l0 = torch.cat(l0s).mean()
            elif self.dp_step >= self.dp_burnin_step:
                l0 = torch.tensor(0)
                if self.dp_step == self.dp_burnin_step:
                    for p in self.dp_mask:
                        alpha = getattr(self.jiant_model.encoder, "ber__" + p.replace(".", "__"))
                        alpha.requires_grad = False
                        _, idx = torch.topk(alpha.flatten().abs(), k = int(float(self.user_mode['diffprun']) * alpha.numel()))
                        self.dp_mask[p].flatten()[idx] = 1.0
                for p in parms:
                    if not p.startswith('encoder.w'):
                        continue
                    p = p.replace("encoder.w__", "").replace("__", ".")
                    w = getattr(self.jiant_model.encoder, "w__" + p.replace(".", "__"))
                    nw = self.jiant_model.ori_pars[p] + self.dp_mask[p] * w
                    node = self.jiant_model
                    pnames = p.split(".")
                    for pname in pnames[:-1]:
                        node = getattr(node, pname)
                    delattr(node, pnames[-1]) 
                    setattr(node, pnames[-1], nw)
            

        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            if 'prompt' in self.user_mode:
                ptsize = int(self.user_mode['prompt'])
                input_ids = batch.input_ids.new_zeros([batch.input_ids.shape[0], ptsize])
                input_ids[:] = torch.arange(ptsize) + self.jiant_model.encoder.embeddings.word_embeddings.num_embeddings
                batch.input_ids = torch.cat([input_ids, batch.input_ids], 1)
                batch.input_mask = torch.cat([torch.ones_like(input_ids), batch.input_mask], 1)
                batch.segment_ids = torch.cat([torch.zeros_like(input_ids), batch.segment_ids], 1)
            elif 'qapp_functorch' in self.user_mode:#functorch
                model = copy.deepcopy(self.jiant_model)
                model.eval()
                func, params, buffers = make_functional_with_buffers(model)
                def compute_loss(params, buffers):
                    y = func(params, buffers, batch=batch, task=task, compute_loss=True)
                    return y['loss']
                grad(grad(compute_loss))(params, buffers)
            elif 'qapp' in self.user_mode:
                if not hasattr(self, 'q_h'):
                    self.q_h = {}
                if not hasattr(self, 'bs_step') or self.bs_step < self.bs_burnin_step:
                    #import copy
                    model = copy.deepcopy(self.jiant_model)
                    model.eval()
                    params, names = make_functional(model)
                    # Make params regular Tensors instead of nn.Parameter
                    params = tuple(p.detach().requires_grad_() for p in params)
                    # your forward function with update
                    def forward(*new_params):
                        # this line replace your for loop
                        load_weights(model, names, new_params)
                        model_output = wrap_jiant_forward(
                            jiant_model=model, batch=batch, task=task, compute_loss=True,
                        )
                        return model_output.loss
                    ones = tuple([torch.ones_like(p) for p in params])
                    hv = hvp(forward, params, ones)[1]
                    hvs = {name : t for name, t in zip(names, hv)}
            elif 'hda_backpack' in self.user_mode:
                from backpack.extensions import BatchDiagHessian, DiagHessian
                from backpack import backpack, extend
                model = copy.deepcopy(self.jiant_model)
                model = extend(model)
                model.eval()
                model_output = wrap_jiant_forward(
                            jiant_model=model, batch=batch, task=task, compute_loss=False,
                        )
                loss_fct = extend(nn.CrossEntropyLoss())
                loss = loss_fct(model_output.logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
                with backpack(DiagHessian(), BatchDiagHessian()):
                    loss = self.complex_backpropagate(
                        loss=loss,
                        gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
                        #retain_graph = retain_graph
                    )
            elif 'hda' in self.user_mode:
                if not hasattr(self, 'hvs'):
                    self.hvs = {}
                if not hasattr(self, 'bs_step') or self.bs_step < self.bs_burnin_step:
                    #import copy
                    model = copy.deepcopy(self.jiant_model)
                    model.eval()
                    params, names = make_functional(model)
                    # Make params regular Tensors instead of nn.Parameter
                    params = tuple(p.detach().requires_grad_() for p in params)
                    # your forward function with update
                    def forward(*new_params):
                        # this line replace your for loop
                        load_weights(model, names, new_params)
                        model_output = wrap_jiant_forward(
                            jiant_model=model, batch=batch, task=task, compute_loss=True,
                        )
                        return model_output.loss
                    rad = tuple([(torch.rand_like(p) > 0.5).float() * 2 - 1 for p in params])
                    N = 10
                    for i in range(N):
                        hv = vhp(forward, params, rad)[1]
                        for name, r, t in zip(names, rad, hv):
                            self.hvs[name] = t*r / N if name not in self.hvs else self.hvs[name] + t*r / N





            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )
            if 'diffprun' in self.user_mode:
                lbd = float(self.user_mode['lambda']) if ('lambda' in self.user_mode) else 1.
                model_output.loss = model_output.loss + lbd * l0
            if 'l2sp' in self.user_mode:
                lbd = float(self.user_mode['l2sp']) if ('l2sp' in self.user_mode and self.user_mode['l2sp'] is not None) else 1.
                parms = dict(self.jiant_model.named_parameters())
                if not hasattr(self, "l2sp_w0"):
                    #import copy
                    self.l2sp_w0 = {}
                    for p in parms:
                        if 'taskmodels_dict' in p: 
                            continue
                        self.l2sp_w0[p] = copy.deepcopy(parms[p].data)
                rs = []
                for p in self.l2sp_w0:
                    rs.append((parms[p] - self.l2sp_w0[p]).flatten()**2)
                model_output.loss = model_output.loss + lbd * torch.cat(rs).mean()
            if 'lnsr' in self.user_mode:
                #import copy
                lbd = float(self.user_mode['lambda']) if ('lambda' in self.user_mode) else 1.
                embbak = self.jiant_model.encoder.embeddings.word_embeddings.weight.data
                self.jiant_model.encoder.embeddings.word_embeddings.weight.data = copy.deepcopy(embbak) + torch.randn_like(embbak) * 0.01
                with torch.no_grad():
                    model_output1 = wrap_jiant_forward(
                        jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
                    )
                self.jiant_model.encoder.embeddings.word_embeddings.weight.data = embbak
                a, b = model_output.other[-1], model_output1.other[-1]
                if type(a) is list:
                    a, b = torch.cat(a), torch.cat(b)
                model_output.loss = model_output.loss + lbd * ((a - b)**2).mean()
            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
                #retain_graph = retain_graph
            )
            loss_val += loss.item()

        if 'bottleneck' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            for p in parms:
                if 'output.dense' not in p and 'taskmodels_dict' not in p: 
                    parms[p].grad = None
                    continue
                elif 'output.dense' in p and 'weight' in p:
                    parms[p].grad[:, int(parms[p].shape[1] * 0.2):] = 0
        elif 'fixptm' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            for p in parms:
                if 'taskmodels_dict' not in p:
                    parms[p].grad = None
        elif 'randmask' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "grad_mask"):
                self.grad_mask = {}
                for p in parms:
                    #if p.startswith('encoder') and ('attention' in p or 'embeddings' in p): # quite good why?
                    if p.startswith('encoder'):
                        self.grad_mask[p] = torch.rand_like(parms[p]) > float(self.user_mode['randmask'])
            for p in parms:
                if p in self.grad_mask and parms[p].grad is not None:
                    parms[p].grad.masked_fill_(self.grad_mask[p], 0.)
        elif 'psearch' in self.user_mode:# DEPRECATED: It still changes all parms
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "ps_mask"):
                self.ps_mask = {}
                self.ps_accu = {}
                self.ps_masked = {}
                self.ps_finished = {}
                self.ps_step = 0
                self.ps_update = 10
                self.ps_r = float(self.user_mode['psearch']) / 2
                for p in parms:
                    self.ps_mask[p] = torch.zeros_like(parms[p]).bool()
                    self.ps_masked[p] = 0
                    self.ps_finished[p] = False
                    self.ps_accu[p] = 0
            for p in parms:
                if parms[p].grad is not None:
                    self.ps_accu[p] = self.ps_accu[p] + parms[p].grad.abs()
                if not self.ps_finished[p] and self.ps_step % self.ps_update == self.ps_update - 1 and self.ps_step > 1 and p.startswith('encoder'):
                    remain = self.ps_accu[p].masked_fill(self.ps_mask[p], float('inf'))
                    size = parms[p].numel()
                    _, idx = torch.topk(-remain.flatten(), k = int(self.ps_r * size))
                    newm = self.ps_masked[p] + len(idx)
                    if newm >= (1 - float(self.user_mode['psearch'])) * size:
                        self.ps_finished[p] = True
                        print("%s : Fixed."%p)
                        continue
                    self.ps_mask[p].flatten()[idx] = True
                if p in self.ps_mask:
                    parms[p].grad.masked_fill_(self.ps_mask[p], 0.)
            self.ps_step += 1
        elif 'bsearch' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "bs_mask"):
                self.bs_burnin_step = 500 if 'burnin' not in self.user_mode else int(self.user_mode['burnin'])
                self.bs_step = 0
                self.bs_mask = {}
                self.bs_accu = {}
                for p in parms:
                    self.bs_mask[p] = torch.ones_like(parms[p]).bool()
                    self.bs_accu[p] = 0
                if "happ" in self.user_mode:
                    self.happ_accu = {}
                    self.happ_prev = {}
                    for p in parms:
                        self.happ_accu[p] = 0
                        self.happ_prev[p] = 0
            self.bs_step += 1
            if self.bs_step < self.bs_burnin_step:
                if 'arand' in self.user_mode:
                    self.jiant_model.eval()
                for p in parms:
                    if parms[p].grad is not None:   
                        if "fisher" in self.user_mode:
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad**2
                        elif "abs" in self.user_mode:
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad.abs()
                        elif "m1" in self.user_mode:
                            self.q_h[p] = hvs[p] if p not in self.q_h else self.q_h[p] + hvs[p]
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad
                        elif "qapp" in self.user_mode:
                            nhv = 1. / hvs[p] / 1000
                            nhv[(nhv > 1.) | (nhv < -1.)] = 0
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad * nhv
                        elif "happ" in self.user_mode:
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad
                            self.happ_accu[p] += (parms[p].grad - (self.happ_prev[p] if p in self.happ_prev else 0)).abs()
                            self.happ_prev[p] = parms[p].grad
                        else:
                            self.bs_accu[p] = self.bs_accu[p] + parms[p].grad
                if "fdm" in self.user_mode:
                    hmodel = copy.deepcopy(self.jiant_model)
                    hparms = dict(hmodel.named_parameters())
                    for p in hparms:
                        hparms[p].data = hparms[p].data * 0.9
                    hmodel_output = wrap_jiant_forward(
                        jiant_model=hmodel, batch=batch, task=task, compute_loss=True,
                    )
                    hloss = self.complex_backpropagate(
                        loss=hmodel_output.loss,
                        gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
                        #retain_graph = retain_graph
                    )
                    if not hasattr(self, 'fdm_hessian'):
                        self.fdm_hessian = {p: 0 for p in parms}
                    hparms = dict(hmodel.named_parameters())
                    for p in parms:
                        hpp = (parms[p].grad - hparms[p].grad) / (0.1 * parms[p].data)
                        hpp[hpp.isnan()] = 0
                        self.fdm_hessian[p] += hpp
                self.optimizer_scheduler.optimizer.zero_grad()
                return
            elif self.bs_step == self.bs_burnin_step:
                for p in parms:
                    if parms[p].grad is None:
                        continue
                    if "vanish" in self.user_mode:
                        _, idx = torch.topk(-self.bs_accu[p].flatten().abs(), k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    elif "m1" in self.user_mode:
                        #score = self.bs_accu[p] * (1. / self.q_h[p]).clamp(min=-100, max=100) / 100
                        score = self.bs_accu[p] * ((1. / self.q_h[p]).sigmoid()*2-1)
                        _, idx = torch.topk(score.flatten().abs(), k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    elif "happ" in self.user_mode:
                        score = self.bs_accu[p].abs() - self.happ_accu[p]
                        score[score.isnan()] = 0
                        _, idx = torch.topk(score.flatten(), k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    elif "fdm" in self.user_mode:
                        #self.fdm_hessian[p][self.fdm_hessian[p].abs() < 1e-3] = 0
                        scale = float(self.user_mode['fdm']) if 'fdm' in self.user_mode else 1.
                        score = (self.bs_accu[p] * ((1. / self.fdm_hessian[p] * scale).sigmoid()*2-1) ).abs()
                        score[score.isnan()] = 0
                        _, idx = torch.topk(score.flatten(), k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    elif 'hda_v1' in self.user_mode:
                        r = float(self.user_mode['hda']) if self.user_mode['hda'] is not None else 0.1
                        ksize = 11#self.hvs[p].numel() // 10
                        h = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(self.hvs[p].flatten(), (ksize // 2, ksize // 2)).unsqueeze(0), ksize, stride = 1).abs()
                        h = h.squeeze(0)
                        m = h[h!=0].mean()
                        h1 = h * r + (1 - r) * m
                        h2 = (1 / h1)
                        s = (h2 < 1).float()
                        h3 = s * h2 + (1 - s) * (1 + h2.log10())
                        print(h3.max(), h3.min(), (h3.max() - h3.min()) / h3.min())
                        score = (self.bs_accu[p].flatten() * h3 ).abs()
                        _, idx = torch.topk(score, k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    elif 'hda' in self.user_mode:
                        r = float(self.user_mode['hda']) if self.user_mode['hda'] is not None else 0.005
                        score = (2 * self.bs_accu[p].flatten().abs().log10() - self.hvs[p].flatten().abs().clamp(0.1).log10() * r)
                        b = self.hvs[p].flatten().abs().clamp(0.1).log10() * r
                        a = 2 * self.bs_accu[p].flatten().abs().log10()
                        #print(b.min().item(), b.max().item(), b.max().item() - b.min().item(), a.max().item())
                        _, idx = torch.topk(score, k = int(float(self.user_mode['bsearch']) * parms[p].numel()))

                        _, idx1 = torch.topk(a, k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                        sidx = set(idx1.tolist())
                        print(len([i for i in idx.tolist() if i in sidx]) / max(len(idx), 1))


                    else:
                        _, idx = torch.topk(self.bs_accu[p].flatten().abs(), k = int(float(self.user_mode['bsearch']) * parms[p].numel()))
                    self.bs_mask[p].flatten()[idx] = False
                if 'arand' in self.user_mode:
                    self.jiant_model.train()
            for p in parms:
                if p in self.bs_mask and p.startswith('encoder') and parms[p].grad is not None:
                    parms[p].grad.masked_fill_(self.bs_mask[p], 0.)
        elif 'magprun' in self.user_mode:# former impabs
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "mag_step"):
                self.mag_burnin_step = 500 if 'burnin' not in self.user_mode else int(self.user_mode['burnin'])
                self.mag_step = 0
            self.mag_step += 1
            if self.mag_step == self.mag_burnin_step and not hasattr(self, "ia_mask"):
                self.ia_mask = {}
                for p in parms:
                    self.ia_mask[p] = torch.ones_like(parms[p]).bool()
                    _, idx = torch.topk(parms[p].abs().flatten(), k = int(float(self.user_mode['magprun']) * parms[p].numel()))
                    self.ia_mask[p].flatten()[idx] = False
            for p in parms:
                if hasattr(self, 'ia_mask') and p in self.ia_mask and p.startswith('encoder') and parms[p].grad is not None:
                    parms[p].grad.masked_fill_(self.ia_mask[p], 0.)
        elif 'impsa' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "isa_mask"):
                self.isa_mask = {}
                plist = []
                for p in parms:
                    self.isa_mask[p] = torch.ones_like(parms[p]).bool()
                    plist.append(parms[p].data.cpu().flatten())
                apars = torch.cat(plist)
                _, idx = torch.topk(apars, k = int(float(self.user_mode['impsa']) * apars.numel()))
                startid = 0
                for p in parms:
                    pids = idx.masked_select((startid <= idx) & (idx < startid + parms[p].numel()))
                    pids -= startid
                    self.isa_mask[p].flatten()[pids] = False
                    startid += parms[p].numel()
                    if pids.numel() == 0 and p.startswith('encoder'):
                        parms[p].requires_grad = False
            for p in parms:
                if p in self.isa_mask and p.startswith('encoder') and parms[p].requires_grad:
                    parms[p].grad.masked_fill_(self.isa_mask[p], 0.)
        elif 'impback' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            if not hasattr(self, "ib_mask"):
                self.ib_burnin_step = 1000
                self.ib_step = 0
                self.ib_mask = {}
                self.ib_weights = {}
                for p in parms:
                    self.ib_mask[p] = torch.ones_like(parms[p]).bool()
                    self.ib_weights[p] = parms[p].detach().cpu()
            self.ib_step += 1
            if self.ib_step == self.ib_burnin_step:
                for p in parms:
                    _, idx = torch.topk(parms[p].abs().flatten(), k = int(float(self.user_mode['impback']) * parms[p].numel()))
                    parms[p].data[:] = self.ib_weights[p]
                    self.ib_mask[p].flatten()[idx] = False
            if self.ib_step >= self.ib_burnin_step:
                for p in parms:
                    if p in self.ib_mask and p.startswith('encoder'):
                        parms[p].grad.masked_fill_(self.ib_mask[p], 0.)
        elif 'bitfit' in self.user_mode:
            parms = dict(self.jiant_model.named_parameters())
            for p in parms:
                if not p.endswith(".bias") and "taskmodels_dict" not in p and parms[p].grad is not None:
                    parms[p].grad[:] = 0
        elif 'gproj' in self.user_mode:
            if not hasattr(self, "gp_step"):
                #import copy
                parms = dict(self.jiant_model.named_parameters())
                self.w0 = copy.deepcopy(parms)
                self.gp_step = 0
                self.gp_mask = {}
                self.gp_burnin_step = 1e100 if 'burnin' not in self.user_mode else int(self.user_mode['burnin'])
                self.gp_gstep = 1 if 'gstep' not in self.user_mode else int(self.user_mode['gstep'])
            self.gp_step += 1
            if self.gp_step % self.gp_gstep == 0 and self.gp_step <= self.gp_burnin_step:
                parms = dict(self.jiant_model.named_parameters())
                for p in parms:
                    if parms[p].grad is None or not p.startswith('encoder'):
                        continue
                    _, idx = torch.topk((parms[p] - self.w0[p]).flatten().abs(), k = int(float(self.user_mode['gproj']) * parms[p].numel()))
                    self.gp_mask[p] = torch.ones_like(parms[p]).bool()
                    self.gp_mask[p].flatten()[idx] = False
                    print("The masked_select error has not been fiexed!!!")
                    parms[p].data.masked_select(self.gp_mask[p])[:] = self.w0[p].masked_select(self.gp_mask[p])
                if self.gp_step == self.gp_burnin_step and 'reset' in self.user_mode:
                    for p in parms:
                        parms[p].data[:] = self.w0[p].data
            elif self.gp_step > self.gp_burnin_step:
                parms = dict(self.jiant_model.named_parameters())
                for p in parms:
                    if p in self.gp_mask and p.startswith('encoder') and parms[p].grad is not None:
                        parms[p].grad.masked_fill_(self.gp_mask[p], 0.)
        elif 'sgpa' in self.user_mode:
            if not hasattr(self, "gp_step"):
                #import copy
                parms = dict(self.jiant_model.named_parameters())
                self.w0 = copy.deepcopy(parms)
                self.gp_step = 0
                self.gp_mask = {}
                self.gp_burnin_step = 1e100 if 'burnin' not in self.user_mode else int(self.user_mode['burnin'])
                self.gp_gstep = 1 if 'gstep' not in self.user_mode else int(self.user_mode['gstep'])
                if 'mmt' in self.user_mode:
                    self.gp_mmt = {}
                    self.gp_mmtr = 0.3 if self.user_mode['mmt'] is None else float(self.user_mode['mmt'])
            self.gp_step += 1
            if self.gp_step > self.gp_burnin_step:
                parms = dict(self.jiant_model.named_parameters())
                for p in parms:
                    if p in self.gp_mask and p.startswith('encoder') and parms[p].grad is not None:
                        parms[p].grad.masked_fill_(self.gp_mask[p], 0.)

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        if 'sgpa' in self.user_mode and self.gp_step <= self.gp_burnin_step: 
            parms = dict(self.jiant_model.named_parameters())
            if self.gp_step % self.gp_gstep == 0:
                for p in parms:
                    if parms[p].grad is None or not p.startswith('encoder'):
                        continue
                    with torch.no_grad():
                        diff = (parms[p] - self.w0[p]).flatten().abs()
                        if "mmt" in self.user_mode:
                            diff = self.gp_mmtr * diff + (1-self.gp_mmtr) * self.gp_mmt[p] if p in self.gp_mmt else diff
                            self.gp_mmt[p] = diff
                    _, idx = torch.topk(diff, k = int(float(self.user_mode['sgpa']) * parms[p].numel()))
                    self.gp_mask[p] = torch.ones_like(parms[p]).bool()
                    self.gp_mask[p].flatten()[idx] = False
                    #parms[p].data.masked_select(self.gp_mask[p])[:] = self.w0[p].masked_select(self.gp_mask[p])
                    parms[p].data[:] = parms[p].data[:] * (~self.gp_mask[p]) + self.w0[p] * self.gp_mask[p]

            if self.gp_step == self.gp_burnin_step and 'reset' in self.user_mode:
                for p in parms:
                    parms[p].data[:] = self.w0[p].data

        train_state.step(task_name=task_name)
        entry = {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            }
        if 'diffprun' in self.user_mode:
            entry["loss_val"] = entry["loss_val"] - l0.item()
            entry["loss_l0"] = l0.item()
        self.log_writer.write_entry(
            "loss_train",
            entry,
        )
        pbar.set_postfix({'loss': loss_val / task_specific_config.gradient_accumulation_steps})

    def run_val(self, task_name_list, use_subset=None, return_preds=False, verbose=True, phase = "val"):
        print("Log Dir:", self.log_writer.tb_writer.logdir)
        evaluate_dict = {}
        val_dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset, phase = phase
        )
        val_labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset, label_phase = phase
        )
        emodel = self.jiant_model
        if 'mixout' in self.user_mode:
            #import copy
            emodel = copy.deepcopy(self.jiant_model)
            parms = dict(emodel.named_parameters())
            for p in parms:
                if not p.startswith("encoder."):
                    continue
                node = emodel.encoder
                node0 = self.encoder0
                pnames = p.split(".")
                for pname in pnames[1:-1]:
                    node = getattr(node, pname)
                    node0 = getattr(node0, pname)
                msk = (torch.rand_like(getattr(node, pnames[-1])) < float(self.user_mode['mixout'])).float()
                nw = (1 - msk) * getattr(node, pnames[-1]) +  msk * getattr(node0, pnames[-1])
                delattr(node, pnames[-1]) 
                setattr(node, pnames[-1], nw)

        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader=val_dataloader_dict[task_name],
                val_labels=val_labels_dict[task_name],
                jiant_model=emodel,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                return_preds=return_preds,
                verbose=verbose,
                tag = phase,#maple
                user_mode = self.user_mode,
            )
        return evaluate_dict

    def run_test(self, task_name_list, verbose=True):
        evaluate_dict = {}
        test_dataloader_dict = self.get_test_dataloader_dict()
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_test(
                test_dataloader=test_dataloader_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                verbose=verbose,
            )
        return evaluate_dict

    def get_train_dataloader_dict(self):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task = self.jiant_task_container.task_dict[task_name]
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            train_batch_size = self.jiant_task_container.task_specific_configs[
                task_name
            ].train_batch_size
            train_dataloader_dict[task_name] = InfiniteYield(
                get_train_dataloader_from_cache(
                    train_cache=train_cache, task=task, train_batch_size=train_batch_size,
                )
            )
        return train_dataloader_dict

    def _get_eval_dataloader_dict(self, phase, task_name_list, use_subset=False):
        val_dataloader_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None,
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, task_name_list, use_subset=False, phase = "val"):
        return self._get_eval_dataloader_dict(
            phase, task_name_list=task_name_list, use_subset=use_subset,
        )

    def get_val_labels_dict(self, task_name_list, use_subset=False, label_phase = "val"):
        val_labels_dict = {}
        for task_name in task_name_list:
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_labels_cache = self.jiant_task_container.task_cache_dict[task_name][label_phase + "_labels"]
            val_labels = val_labels_cache.get_all()
            if use_subset:
                val_labels = val_labels[: task_specific_config.eval_subset_num]
            val_labels_dict[task_name] = val_labels
        return val_labels_dict

    def get_test_dataloader_dict(self):
        return self._get_eval_dataloader_dict(
            task_name_list=self.jiant_task_container.task_run_config.test_task_list,
            phase=PHASE.TEST,
        )

    def complex_backpropagate(self, loss, gradient_accumulation_steps, retain_graph = False):
        return complex_backpropagate(
            loss=loss,
            optimizer=self.optimizer_scheduler.optimizer,
            model=self.jiant_model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
            retain_graph = retain_graph
        )

    def get_runner_state(self):
        # TODO: Add fp16  (issue #1186)
        state = {
            "model": torch_utils.get_model_for_saving(self.jiant_model).state_dict(),
            "optimizer": self.optimizer_scheduler.optimizer.state_dict(),
        }
        return state

    def load_state(self, runner_state):
        torch_utils.get_model_for_saving(self.jiant_model).load_state_dict(runner_state["model"])
        self.optimizer_scheduler.optimizer.load_state_dict(runner_state["optimizer"])


class CheckpointSaver:
    def __init__(self, metadata, save_path):
        self.metadata = metadata
        self.save_path = save_path

    def save(self, runner_state: dict, metarunner_state: dict):
        to_save = {
            "runner_state": runner_state,
            "metarunner_state": metarunner_state,
            "metadata": self.metadata,
        }
        torch_utils.safe_save(to_save, self.save_path)


def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    return_preds=False,
    verbose=True,
    tag="Val",
    user_mode = None,
):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    jiant_model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, {tag})", verbose=verbose)
    ):
        batch = batch.to(device)
        if user_mode is not None and 'prompt' in user_mode:
            ptsize = int(user_mode['prompt'])
            input_ids = batch.input_ids.new_zeros([batch.input_ids.shape[0], ptsize])
            input_ids[:] = torch.arange(ptsize) + jiant_model.encoder.embeddings.word_embeddings.num_embeddings
            batch.input_ids = torch.cat([input_ids, batch.input_ids], 1)
            batch.input_mask = torch.cat([torch.ones_like(input_ids), batch.input_mask], 1)
            batch.segment_ids = torch.cat([torch.zeros_like(input_ids), batch.segment_ids], 1)
        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
            batch_metadata=batch_metadata,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps
    tokenizer = (
        jiant_model.tokenizer
        if not torch_utils.is_data_parallel(jiant_model)
        else jiant_model.module.tokenizer
    )
    output = {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        ),
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
    return output


def run_test(
    test_dataloader,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    verbose=True,
    return_preds=True,
):
    if not local_rank == -1:
        return
    jiant_model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits, batch_loss=0, batch=batch, batch_metadata=batch_metadata,
        )
    output = {
        "accumulator": eval_accumulator,
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
    return output
