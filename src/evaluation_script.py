# Radial Modified Evaluation Script 
import transformer.manage as DT_manager
import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator
import json, pathlib, math, random
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import DecisionTransformerConfig, GPT2Config, GPT2LMHeadModel
from transformer.art import AutonomousRendezvousTransformer # get actual model
import os, sys
import shutil
from torch.optim import AdamW
from accelerate import Accelerator
from radialSelfAttention import radial_swap
from transformer.manage import torch_check_koz_constraint
import copy
from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *

# rewrite model evaluation
def get_DT_model(model_name: str,
                 train_loader=None,
                 eval_loader=None,
                 *,
                 w0: int = 16,
                 keep_dense: int = 2,
                 lora_subdir: str = "lora_step_99001"): # can change default depending on how long we fine tune
    """
    model_name   : folder under transformer/saved_files/checkpoints/
                   that contains pytorch_model.bin
    lora_subdir  : sub-folder that contains adapter_model.bin + adapter_config.json
    """
    ckpt_root = Path("transformer/saved_files/checkpoints") / model_name
    base_ckpt = "transformer/saved_files/checkpoints/checkpoint_rtn_art/pytorch_model.bin"
    lora_dir  = ckpt_root / lora_subdir

    cfg = DecisionTransformerConfig(
        state_dim = train_loader.dataset.n_state,
        act_dim   = train_loader.dataset.n_action,
        hidden_size = 384,
        max_ep_len  = 100,
        n_layer = 6,
        n_head  = 6,
    )

    base_model = AutonomousRendezvousTransformer(cfg)
    base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))

    radial_swap(base_model)

    model = PeftModel.from_pretrained(base_model, lora_dir, is_trainable=False)

    # need fp16 as flash attn doesnt support fp32
    return model.half().to(device).eval()

DT_manager.get_DT_model = get_DT_model # my function now
# adjust inference function to fp16 too
def torch_model_inference_dyn(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)

    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample

    # swap to fp16
    timesteps_i      = timesteps_i.view(1, n_time).long().to(device)
    attention_mask_i = attention_mask_i.view(1, n_time).long().to(device)
    dtype = next(model.parameters()).dtype          # fp16
    states_i = states_i.to(device, dtype=dtype)
    rtgs_i   = rtgs_i.to(device, dtype=dtype)
    ctgs_i   = ctgs_i.view(1, n_time, 1).to(device, dtype=dtype)

    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]

    # For loop trajectory generation
    for t in np.arange(n_time):

        ##### Dynamics inference
        # Compute action pred for dynamics model
        with torch.no_grad():
            output_dyn = model(
                states=states_dyn[:,:t+1,:].contiguous().to(dtype),
                actions=actions_dyn[:,:t+1,:].contiguous().to(dtype),
                rewards=None,
                returns_to_go=rtgs_dyn[:,:t+1,:].contiguous().to(dtype),
                constraints_to_go=ctgs_dyn[:,:t+1,:].contiguous().to(dtype),
                timesteps=timesteps_i[:,:t+1],
                attention_mask=attention_mask_i[:,:t+1],
                return_dict=False,
            )
            (_, action_preds_dyn) = output_dyn

        action_dyn_t = action_preds_dyn[0,t]
        actions_dyn[:,t,:] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable
        if t != n_time-1:
            roe_dyn[:,t+1] = stm[:,:,t] @ (roe_dyn[:,t] + cim[:,:,t] @ dv_dyn[:,t])
            rtn_dyn[:,t+1] = psi[:,:,t+1] @ roe_dyn[:,t+1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            states_dyn[:,t+1,:] = states_dyn_norm

            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:,t+1], t+1)
            ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - viol_dyn
            actions_dyn[:,t+1,:] = 0

        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn' : rtn_dyn.cpu().numpy(),
        'roe_dyn' : roe_dyn.cpu().numpy(),
        'dv_dyn' : dv_dyn.cpu().numpy(),
        'time_orb' : time_orb
    }

    return DT_trajectory, runtime_DT

DT_manager.torch_model_inference_dyn = torch_model_inference_dyn # mine again

# Evaluation Script

root_folder = ""

#print(sys.path)

# Initializations
warmstart = 'both' # 'cvx'/'transformer'/'both'
scenario_test_dataset = True
state_representation = 'rtn' # 'roe'/'rtn'
dataset_to_use = 'both' # 'scp'/'cvx'/'both'
transformer_ws = 'dyn' # 'dyn'/'ol'
transformer_model_name = 'checkpoint_rtn_art_train'
select_idx = True # set to True to manually select a test trajectory via its index (idx)
idx = 18111 # index of the test trajectory (e.g., idx = 18111)
exclude_scp_cvx = False
exclude_scp_DT = False

# Scenario sampling
if not scenario_test_dataset:
    # Transfer horizon (orbits)
    hrz = 2
    # Initial relative orbit
    da = 0 # [m]
    dlambda = 75 # [m]
    de = 1/E_koz.item((0,0))+10
    di = 1/E_koz.item((2,2))+10
    ph_de = np.pi/2 + 0*np.pi/180; # [m]
    ph_di = np.pi/2 + 0*np.pi/180; # [m]
    state_roe_0 = np.array([da, dlambda, de*np.cos(ph_de), de*np.sin(ph_de), di*np.cos(ph_di), di*np.sin(ph_di)]).reshape((6,))
    relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)
else:
    # Get the datasets and loaders from the torch data
    datasets, dataloaders = DT_manager.get_train_val_test_data(state_representation, dataset_to_use, "checkpoint_rtn_art")
    train_loader, eval_loader, test_loader = dataloaders

    # Sample from test dataset
    if select_idx:
        test_sample = test_loader.dataset.getix(idx)
    else:
        test_sample = next(iter(test_loader))
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = test_sample

    print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
    data_stats = test_loader.dataset.data_stats

    hrz = horizons.item()
    if state_representation == 'roe':
        state_roe_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    elif state_representation == 'rtn':
        state_rtn_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_roe_0 = map_rtn_to_roe(state_rtn_0, np.array(oe[0, :, 0]))
    # relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)

# Dynamics Matrices Precomputations
stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dynamics_roe_optimization(oe_0_ref, t_0, hrz, n_time_rpod)

# Build the oe vector including the target instant
oe_hrz_trg = np.append(oe_hrz,np.array([oe_0_ref.item(0), oe_0_ref.item(1), oe_0_ref.item(2), oe_0_ref.item(3), oe_0_ref.item(4), oe_0_ref.item(5) + n_ref*(time_hrz[-1]+dt_hrz-t_0)]).reshape((6,1)),1)
time_hrz_trg = np.append(time_hrz, time_hrz[-1]+dt_hrz)

# Warmstarting and optimization
if warmstart == 'cvx' or warmstart == 'both':
    # Solve Convex Problem
    runtime_cvx0 = time.time()
    states_roe_cvx, actions_cvx, feas_cvx = ocp_cvx(stm_hrz, cim_hrz, psi_hrz, state_roe_0, n_time_rpod)
    runtime_cvx = time.time() - runtime_cvx0
    print('CVX cost:', la.norm(actions_cvx, axis=0).sum())
    print('CVX runtime:', runtime_cvx)
    states_roe_cvx_trg = np.append(states_roe_cvx, (states_roe_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_cvx[:,-1])).reshape((6,1)), 1)
    states_roe_ws_cvx = states_roe_cvx # set warm start
    states_rtn_ws_cvx = roe_to_rtn_horizon(states_roe_cvx_trg, oe_hrz_trg, n_time_rpod+1)
    # Evaluate Constraint Violation
    constr_cvx, constr_viol_cvx = check_koz_constraint(states_rtn_ws_cvx, n_time_rpod+1)
    # Solve SCP
    states_roe_scp_cvx, actions_scp_cvx, feas_scp_cvx, iter_scp_cvx , J_vect_scp_cvx, runtime_scp_cvx = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_cvx, n_time_rpod)
    if states_roe_scp_cvx is None:
        exclude_scp_cvx = True
        print('No scp-cvx solution!')
    else:
        print('SCP cost:', la.norm(actions_scp_cvx, axis=0).sum())
        print('J vect', J_vect_scp_cvx)
        print('SCP runtime:', runtime_scp_cvx)
        print('CVX+SCP runtime:', runtime_cvx+runtime_scp_cvx)
        states_roe_scp_cvx_trg = np.append(states_roe_scp_cvx, (states_roe_scp_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_cvx[:,-1])).reshape((6,1)), 1)
        states_rtn_scp_cvx = roe_to_rtn_horizon(states_roe_scp_cvx_trg, oe_hrz_trg, n_time_rpod+1)
        constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_rtn_scp_cvx, n_time_rpod+1)

if warmstart == 'transformer' or warmstart == 'both':

    # Import the Transformer
    model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader) # can add optionals here
    inference_func = DT_manager.torch_model_inference_dyn
    print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+'dyn idek'+'()')
    rtg = la.norm(actions_cvx, axis=0).sum()
    DT_trajectory, runtime_DT = inference_func(model, test_loader, test_sample, stm_hrz, cim_hrz, psi_hrz, state_representation, rtg_perc=1., ctg_perc=0., rtg=rtg)
    states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws]# set warm start
    # states_rtn_ws_DT = DT_trajectory['rtn_' + transformer_ws]
    actions_rtn_ws_DT = DT_trajectory['dv_' + transformer_ws]
    states_roe_DT_trg = np.append(states_roe_ws_DT, (states_roe_ws_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_rtn_ws_DT[:,-1])).reshape((6,1)), 1)
    states_rtn_ws_DT = roe_to_rtn_horizon(states_roe_DT_trg, oe_hrz_trg, n_time_rpod+1)

    # void nans
    nan_mask = np.isnan(actions_rtn_ws_DT)
    actions_rtn_ws_DT[nan_mask] = 0.0
    art_cost = la.norm(actions_rtn_ws_DT, axis=0).sum()
    print('ART cost:', art_cost)
    print('ART runtime:', runtime_DT)
    constr_DT, constr_viol_DT = check_koz_constraint(states_rtn_ws_DT, n_time_rpod+1)

    # Solve SCP
    states_roe_scp_DT, actions_scp_DT, feas_scp_DT, iter_scp_DT, J_vect_scp_DT, runtime_scp_DT = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_DT, n_time_rpod)
    if states_roe_scp_DT is None:
        exclude_scp_DT = True
        print('No scp-DT solution!')
    else:
        scp_cost = la.norm(actions_scp_DT, axis=0).sum()
        print('SCP cost:', scp_cost)
        print('J vect', J_vect_scp_DT)
        states_roe_scp_DT_trg = np.append(states_roe_scp_DT, (states_roe_scp_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_DT[:,-1])).reshape((6,1)), 1)
        states_rtn_scp_DT = roe_to_rtn_horizon(states_roe_scp_DT_trg, oe_hrz_trg, n_time_rpod+1)
        constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_rtn_scp_DT, n_time_rpod+1)

# Plotting
plt.style.use('seaborn-v0_8-colorblind')
relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)
t_ws_show = dock_wyp_sample

# 3D position trajectory'
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(projection='3d')
ax1.view_init(elev=15, azim=-60, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax1.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k--', linewidth=2.5, label='CVX')
    if not exclude_scp_cvx:
        p2 = ax1.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=3, label='SCP-CVX') # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax1.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b--', linewidth=2.5, label='ART') # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax1.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=3, label='SCP-ART') #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax1.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=2.5, label='way-point')
pell = ax1.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3, label='keep-out-zone')
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax1.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7, label='approach cone')
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax1.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5, label='initial rel. orbit')
#if not exclude_scp_cvx:
p4 = ax1.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=1.5, label='$t_0$')
p5 = ax1.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=1.5, label='docking port')
ax1.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax1.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax1.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax1.tick_params(axis='y', labelcolor='k', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='z', labelsize=15)
ax1.set_xticks(np.linspace(-200, 100, 4))
ax1.set_yticks(np.linspace(-100, 100, 3))
# ax.grid(True)
#ax1.legend(loc='upper left')
ax1.set_box_aspect(aspect=None, zoom=0.9)
plt.tight_layout()
# plt.subplots_adjust(wspace=0.05)
handles1, labels1 = ax1.get_legend_handles_labels()
first_legend = plt.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.85),
          ncol=4, fancybox=True, shadow=True, fontsize=15)
plt.savefig(root_folder + 'optimization/saved_files/plots/pos_3d.png', dpi = 600, bbox_inches='tight')


# 3D position trajectory'
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.view_init(elev=6, azim=-64, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax1.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k--', linewidth=2.5, label='CVX')
    if not exclude_scp_cvx:
        p2 = ax1.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=0) # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax1.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b--', linewidth=2.5, label='ART') # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax1.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=0) #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax1.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=2.5)
pell = ax1.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3)
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax1.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7)
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax1.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5)
#if not exclude_scp_cvx:
p4 = ax1.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=1.5)
p5 = ax1.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=1.5)
ax1.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax1.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax1.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax1.tick_params(axis='y', labelcolor='k', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='z', labelsize=15)
ax1.set_yticks(np.linspace(-100, 100, 3))
# ax1.set_xlim([-200,200])
# ax1.set_ylim([-300,300])
# ax1.set_zlim([-100,100])
ax1.set_box_aspect(aspect=None, zoom=0.945)
# ax.grid(True)
#ax1.legend(loc='upper left')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.view_init(elev=6, azim=-64, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax2.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k-', linewidth=0)
    if not exclude_scp_cvx:
        p2 = ax2.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=2.5, label='SCP-CVX') # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax2.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b-', linewidth=0) # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax2.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=2.5, label='SCP-ART') #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax2.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=1.5, label='way-point')
pell = ax2.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3, label='keep-out-zone')
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax2.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7, label='approach cone')
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax2.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5, label='initial rel. orbit')
p4 = ax2.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=2.5, label='$t_0$')
p5 = ax2.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=2.5, label='docking port')
ax2.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax2.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax2.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax2.tick_params(axis='y', labelcolor='k', labelsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='z', labelsize=15)
ax2.set_yticks(np.linspace(-100, 100, 3))
# ax2.set_xlim([-200,200])
# ax2.set_ylim([-300,300])
# ax2.set_zlim([-100,100])
ax2.set_box_aspect(aspect=None, zoom=0.945)
plt.tight_layout(pad=1.0, w_pad=0.2)
# plt.subplots_adjust(wspace=-0.01)
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
first_legend = plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(-0., 0.85),
          ncol=5, fancybox=True, shadow=True, fontsize=15)
ax2.add_artist(first_legend)
#ax1.legend(loc='lower center', bbox_to_anchor=(0.8, 0.9),
#          ncol=4, fancybox=True, shadow=True, fontsize=8, zorder=1)
plt.savefig(root_folder + 'optimization/saved_files/plots/pos_3d_split.png', dpi = 600, bbox_inches='tight')

# Constraint satisfaction
plt.figure(figsize=(6,4))
if warmstart == 'cvx' or warmstart == 'both':
    plt.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_cvx[:t_ws_show], 'k--', linewidth=1.5, label='CVX')
    if not exclude_scp_cvx:
        plt.plot(time_hrz_trg/period_ref, constr_scp_cvx, 'k-', linewidth=1.8, label='SCP-CVX')

if warmstart == 'transformer' or warmstart == 'both':
    plt.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_DT[:t_ws_show], 'b--', linewidth=1.5, label='ART')
    if not exclude_scp_DT:
        plt.plot(time_hrz_trg/period_ref, constr_scp_DT, 'b-', linewidth=1.8, label='SCP-ART')

plt.plot(time_hrz_trg/period_ref, np.ones(n_time_rpod+1), 'r-', linewidth=1.8, label='koz')
plt.xlabel('time [orbits]', fontsize=15)
plt.ylabel('keep-out-zone constraint [-]', fontsize=15)
# plt.grid(True)
plt.xlim([-0.1,3])
plt.ylim([-0.1,9])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + 'optimization/saved_files/plots/koz_constr.png', dpi = 600)

fig, ax = plt.subplots(figsize=(6,4))
if warmstart == 'cvx' or warmstart == 'both':
    ax.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_cvx[:t_ws_show], 'k--', linewidth=1.8, label='CVX')
    if not exclude_scp_cvx:
        ax.plot(time_hrz_trg/period_ref, constr_scp_cvx, 'k-', linewidth=2, label='SCP-CVX')

if warmstart == 'transformer' or warmstart == 'both':
    ax.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_DT[:t_ws_show], 'b--', linewidth=1.8, label='ART')
    if not exclude_scp_DT:
        ax.plot(time_hrz_trg/period_ref, constr_scp_DT, 'b-', linewidth=2, label='SCP-ART')

ax.plot(time_hrz_trg/period_ref, np.ones(n_time_rpod+1), 'r-', linewidth=1.5, label='koz')
ax.fill_between([0, (time_hrz_trg/period_ref)[-1]], [0, 0], [1, 1], alpha=0.15, color='red')
ax.set_xlabel('Time [orbits]', fontsize=16, linespacing=1.5)
ax.set_ylabel('Keep-out-zone constraint [-]', fontsize=16)
ax.tick_params(axis='y', labelcolor='k', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
# plt.grid(True)
ax.set_xlim([-0.1,3])
ax.set_ylim([-0.1,9])
plt.legend(loc='best', fontsize=15)
plt.tight_layout()
plt.savefig(root_folder + 'optimization/saved_files/plots/koz_constr_v2.png', dpi = 600, bbox_inches='tight')

# ROE plots

# # ROE space
# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[1,:t_ws_show], states_roe_cvx_trg[0,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[1, :], states_roe_scp_cvx_trg[0,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[1,:t_ws_show], states_roe_DT_trg[0,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[1, :], states_roe_scp_DT_trg[0,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta \lambda$ [m]')
# plt.ylabel('$a \delta a$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe12.png')

# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[2,:t_ws_show], states_roe_cvx_trg[3,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[2, :], states_roe_scp_cvx_trg[3,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[2,:t_ws_show], states_roe_DT_trg[3,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[2, :], states_roe_scp_DT_trg[3,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta e_x$ [m]')
# plt.ylabel('$a \delta e_y$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe34.png')

# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[4,:t_ws_show], states_roe_cvx_trg[5,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[4, :], states_roe_scp_cvx_trg[5,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[4,:t_ws_show], states_roe_DT_trg[5,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[4, :], states_roe_scp_DT_trg[5,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta i_x$ [m]')
# plt.ylabel('$a \delta i_y$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe56.png')

# ROE vs time
plot_orb_time = True
plt.figure(figsize=(12,8)) #figsize=(20,5)
for j in range(6):
    plt.subplot(3,2,j+1)
    if warmstart == 'cvx' or warmstart == 'both':
        plt.plot(time_hrz_trg[:t_ws_show]/period_ref, states_roe_cvx_trg[j,:t_ws_show], 'k--', linewidth=1.5, label='CVX')
        if not exclude_scp_cvx:
            plt.plot(time_hrz_trg/period_ref, states_roe_scp_cvx_trg[j,:], 'k-', linewidth=1.8, label='SCP-CVX')
    if warmstart == 'transformer' or warmstart == 'both':
        plt.plot(time_hrz_trg[:t_ws_show]/period_ref, states_roe_DT_trg[j,:t_ws_show], 'b--', linewidth=1.5, label='ART')
        if not exclude_scp_DT:
            plt.plot(time_hrz_trg/period_ref, states_roe_scp_DT_trg[j,:], 'b-', linewidth=1.8, label='SCP-ART')
    if j == 0:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta a$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-60,30])
        plt.legend(loc='best', fontsize=11.5)
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-50,50,25))
    elif j == 1:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta \lambda$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-100,200])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-100,250,50))
        # plt.legend(loc='best')
    elif j == 2:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta e_x$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-30,20])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-25,25,10))
        # plt.legend(loc='best')
    elif j == 3:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta e_y$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-5,100])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(0,120,20))
        # plt.legend(loc='best')
    elif j == 4:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta i_x$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-6,6])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(np.arange(-5,7.5,2.5))
        # plt.legend(loc='best')
    elif j == 5:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta i_y$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-5,140])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(0,150,25))
        # plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + 'optimization/saved_files/plots/roe_vs_time.png', dpi = 600, bbox_inches='tight')

# Control
plt.figure(figsize=(12,8)) #figsize=(20,5)
for j in range(3):
    plt.subplot(1,3,j+1)
    plt.stem(time_hrz[:t_ws_show]/period_ref, actions_cvx[j,:t_ws_show]*1000., 'k--', markerfmt='D', label='CVX')
    plt.stem(time_hrz/period_ref, actions_scp_cvx[j,:]*1000., 'k-', label='SCP-CVX')
    plt.stem(time_hrz[:t_ws_show]/period_ref, actions_rtn_ws_DT[j,:t_ws_show]*1000., 'b--', markerfmt='D', label='ART')
    plt.stem(time_hrz/period_ref, actions_scp_DT[j,:]*1000., 'b-', label='SCP-ART')
    if j == 0:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_r$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-15,25])
        plt.legend(loc='best', fontsize=15)
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.xticks(range(0,4,1))
    elif j == 1:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_t$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-25,25])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-25,30,5))
        plt.xticks(range(0,4,1))
        # plt.legend(loc='best')
    elif j == 2:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_n$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-60,60])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.xticks(range(0,4,1))
        # plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + 'optimization/saved_files/plots/delta_v.png', dpi = 600, bbox_inches='tight')

# # Cost
# if not (exclude_scp_cvx or exclude_scp_DT):
#     plt.figure()
#     max_it = max(iter_scp_cvx, iter_scp_DT)
#     for i in range(max_it):
#         if i >= iter_scp_cvx:
#             J_vect_scp_cvx[i] = J_vect_scp_cvx[iter_scp_cvx-1]
#         elif i >= iter_scp_DT:
#             J_vect_scp_DT[i] = J_vect_scp_DT[iter_scp_DT-1]
#     if warmstart == 'cvx' or warmstart == 'both':
#         plt.plot(J_vect_scp_cvx[:max_it]*1000., 'b--', marker='o', linewidth=1.5, label='SCP-CVX')

#     if warmstart == 'transformer' or warmstart == 'both':
#         plt.plot(J_vect_scp_DT[:max_it]*1000., 'g--', marker='o', linewidth=1.5, label='SCP-ART')

#     plt.xlabel('Iterations [-]')
#     plt.ylabel('Cost [m/s]')
#     # plt.grid(True)
#     plt.legend(loc='best')
#     plt.savefig(root_folder + '/optimization/saved_files/plots/cost.png')

# io.savemat('plot_data.mat', dict(states_rtn_ws_cvx=states_rtn_ws_cvx, states_rtn_scp_cvx=states_rtn_scp_cvx, states_rtn_ws_DT=states_rtn_ws_DT, states_rtn_scp_DT=states_rtn_scp_DT, dock_wyp=dock_wyp, x_ell=x_ell, y_ell=y_ell, z_ell=z_ell, t_cone=t_cone, n_cone=n_cone, r_cone=r_cone))

plt.show()
# comparison to base costs:
base_ovrl = 0.20998219236290414
base_art = 0.2724136

print(f"radial/base art cost: {art_cost/base_art}")
print(f"radial/base art+scp overall cost: {scp_cost/base_ovrl}")