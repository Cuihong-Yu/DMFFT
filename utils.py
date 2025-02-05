import numpy

import torch
import torch.fft as fft

from diffusers.utils import is_torch_version

from typing import Any, Dict, Optional, Tuple


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2

    if crow - threshold < 0:
        top = 0
    else:
        top = crow - threshold

    if ccol - threshold < 0:
        left = 0
    else:
        left = ccol - threshold

    mask[..., top:crow + threshold, left:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1))
    x_filtered = x_filtered.real
    
    x_filtered = x_filtered.type(dtype)

    return x_filtered


def fourier_solo(x, global_scale=1.0, freq_threshold=0, lf_scale=1.0, hf_scale=1.0, amplitude_scale=1.0, phase_scale=1.0, blend_type=0):
    dtype = x.dtype

    x = x.type(torch.float32)

    x = x * global_scale

    if blend_type == -1:
        x = x.type(dtype)

        return x

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    mask[...] = hf_scale

    crow, ccol = H // 2, W // 2

    if crow - freq_threshold < 0:
        top = 0
    else:
        top = crow - freq_threshold

    if ccol - freq_threshold < 0:
        left = 0
    else:
        left = ccol - freq_threshold

    mask[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_scale
    x_freq = x_freq * mask

    amplitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)

    if blend_type == 0:
        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale
    elif blend_type == 1:
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] * amplitude_scale
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] * phase_scale
    elif blend_type == 2:
        lf_amplitude = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold]
        lf_phase = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold]

        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale

        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_amplitude
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_phase
    elif blend_type == 3:
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] * amplitude_scale

        lf_phase = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold]
        phase = phase * phase_scale
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_phase
    elif blend_type == 4:
        lf_amplitude = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold]
        amplitude = amplitude * amplitude_scale
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_amplitude

        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] * phase_scale

    reals = amplitude * torch.cos(phase)
    images = amplitude * torch.sin(phase)

    x_freq = torch.complex(reals, images)

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)

    return x_filtered


def register_tune_upblock2d(model, types=0,
                            k1=0.5, b1=1.2, t1=1, s1=0.9,
                            k1_1=0.5, b1_1=1.2, t1_1=1, s1_1=0.9,
                            k2=0.5, b2=1.4, t2=1, s2=0.2,
                            k2_1=0.5, b2_1=1.4, t2_1=1, s2_1=0.2,
                            g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
                            blend1=0, blend2=0, blend1_1=0, blend2_1=0,
                            a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
                            p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
                            skips=0, tunes=0):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            skipping = 0
            finetune = 0

            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = hidden_states[:, :int(hidden_states.shape[1] * self.k1)] * self.b1
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = hidden_states[:, :int(hidden_states.shape[1] * self.k2)] * self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = hidden_states[:, :int(hidden_states.shape[1] * self.k1)] * self.b1
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] * self.s1

                        finetune = finetune + 1
                elif self.types == 3:
                    hidden_states_numpy = hidden_states.to('cpu').detach().numpy()
                    hidden_states_avg = numpy.mean(hidden_states_numpy, axis=1)
                    hidden_states_max = numpy.max(hidden_states_avg, axis=(1, 2))
                    hidden_states_min = numpy.min(hidden_states_avg, axis=(1, 2))

                    if hidden_states.shape[1] == 1280:
                        hidden_states_relation = numpy.zeros((hidden_states.shape[0], int(hidden_states.shape[1] * self.k1), hidden_states.shape[2], hidden_states.shape[3]))

                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (hidden_states_avg[index, ...] - hidden_states_min[index]) / (hidden_states_max[index] - hidden_states_min[index])

                        hidden_states_alpha = 1.0 + (self.b1 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to('cuda')

                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = hidden_states[:, :int(hidden_states.shape[1] * self.k1)] * hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states_relation = numpy.zeros((hidden_states.shape[0], int(hidden_states.shape[1] * self.k2), hidden_states.shape[2], hidden_states.shape[3]))

                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (hidden_states_avg[index, ...] - hidden_states_min[index]) / (hidden_states_max[index] - hidden_states_min[index])

                        hidden_states_alpha = 1.0 + (self.b2 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to('cuda')

                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = hidden_states[:, :int(hidden_states.shape[1] * self.k2)] * hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        if self.k1 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k1)], global_scale=self.g1, freq_threshold=self.t1, lf_scale=self.b1, hf_scale=self.s1, amplitude_scale=self.a1, phase_scale=self.p1, blend_type=self.blend1)

                        if self.k1_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)], global_scale=self.g1_1, freq_threshold=self.t1_1, lf_scale=self.b1_1, hf_scale=self.s1_1, amplitude_scale=self.a1_1, phase_scale=self.p1_1, blend_type=self.blend1_1)

                        finetune = finetune + 1
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k1)], global_scale=self.g1, freq_threshold=self.t1, lf_scale=self.b1, hf_scale=self.s1, amplitude_scale=self.a1, phase_scale=self.p1, blend_type=self.blend1)

                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)], global_scale=self.g1_1, freq_threshold=self.t1_1, lf_scale=self.b1_1, hf_scale=self.s1_1, amplitude_scale=self.a1_1, phase_scale=self.p1_1, blend_type=self.blend1_1)

                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k2)], global_scale=self.g2, freq_threshold=self.t2, lf_scale=self.b2, hf_scale=self.s2, amplitude_scale=self.a2, phase_scale=self.p2, blend_type=self.blend2)

                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)], global_scale=self.g2_1, freq_threshold=self.t2_1, lf_scale=self.b2_1, hf_scale=self.s2_1, amplitude_scale=self.a2_1, phase_scale=self.p2_1, blend_type=self.blend2_1)

                        finetune = finetune + 1

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)

                skipping = skipping + 1

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states
        
        return forward
    
    for i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)

            setattr(upsample_block, 'types', types)

            setattr(upsample_block, 'k1', k1)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 't1', t1)
            setattr(upsample_block, 's1', s1)

            setattr(upsample_block, 'k2', k2)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 't2', t2)
            setattr(upsample_block, 's2', s2)

            setattr(upsample_block, 'k1_1', k1_1)
            setattr(upsample_block, 'b1_1', b1_1)
            setattr(upsample_block, 't1_1', t1_1)
            setattr(upsample_block, 's1_1', s1_1)

            setattr(upsample_block, 'k2_1', k2_1)
            setattr(upsample_block, 'b2_1', b2_1)
            setattr(upsample_block, 't2_1', t2_1)
            setattr(upsample_block, 's2_1', s2_1)

            setattr(upsample_block, 'g1', g1)
            setattr(upsample_block, 'g2', g2)
            setattr(upsample_block, 'g1_1', g1_1)
            setattr(upsample_block, 'g2_1', g2_1)

            setattr(upsample_block, 'blend1', blend1)
            setattr(upsample_block, 'blend2', blend2)
            setattr(upsample_block, 'blend1_1', blend1_1)
            setattr(upsample_block, 'blend2_1', blend2_1)

            setattr(upsample_block, 'a1', a1)
            setattr(upsample_block, 'a2', a2)
            setattr(upsample_block, 'a1_1', a1_1)
            setattr(upsample_block, 'a2_1', a2_1)

            setattr(upsample_block, 'p1', p1)
            setattr(upsample_block, 'p2', p2)
            setattr(upsample_block, 'p1_1', p1_1)
            setattr(upsample_block, 'p2_1', p2_1)

            setattr(upsample_block, 'skips', skips)
            setattr(upsample_block, 'tunes', tunes)


def register_tune_crossattn_upblock2d(model, types=0,
                                      k1=0.5, b1=1.2, t1=1, s1=0.9,
                                      k1_1=0.5, b1_1=1.2, t1_1=1, s1_1=0.9,
                                      k2=0.5, b2=1.4, t2=1, s2=0.2,
                                      k2_1=0.5, b2_1=1.4, t2_1=1, s2_1=0.2,
                                      g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
                                      blend1=0, blend2=0, blend1_1=0, blend2_1=0,
                                      a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
                                      p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
                                      skips=0, tunes=0):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            skipping = 0
            finetune = 0

            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = hidden_states[:, :int(hidden_states.shape[1] * self.k1)] * self.b1
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = hidden_states[:, :int(hidden_states.shape[1] * self.k2)] * self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = hidden_states[:, :int(hidden_states.shape[1] * self.k2)] * self.b2
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] * self.s2

                        finetune = finetune + 1
                elif self.types == 3:
                    hidden_states_numpy = hidden_states.to('cpu').detach().numpy()
                    hidden_states_avg = numpy.mean(hidden_states_numpy, axis=1)
                    hidden_states_max = numpy.max(hidden_states_avg, axis=(1, 2))
                    hidden_states_min = numpy.min(hidden_states_avg, axis=(1, 2))

                    if hidden_states.shape[1] == 1280:
                        hidden_states_relation = numpy.zeros((hidden_states.shape[0], int(hidden_states.shape[1] * self.k1), hidden_states.shape[2], hidden_states.shape[3]))

                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (hidden_states_avg[index, ...] - hidden_states_min[index]) / (hidden_states_max[index] - hidden_states_min[index])

                        hidden_states_alpha = 1.0 + (self.b1 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to('cuda')

                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = hidden_states[:, :int(hidden_states.shape[1] * self.k1)] * hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states_relation = numpy.zeros((hidden_states.shape[0], int(hidden_states.shape[1] * self.k2), hidden_states.shape[2], hidden_states.shape[3]))

                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (hidden_states_avg[index, ...] - hidden_states_min[index]) / (hidden_states_max[index] - hidden_states_min[index])

                        hidden_states_alpha = 1.0 + (self.b2 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to('cuda')

                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = hidden_states[:, :int(hidden_states.shape[1] * self.k2)] * hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        if self.k2 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k2)], global_scale=self.g2, freq_threshold=self.t2, lf_scale=self.b2, hf_scale=self.s2, amplitude_scale=self.a2, phase_scale=self.p2, blend_type=self.blend2)

                        if self.k2_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)], global_scale=self.g2_1, freq_threshold=self.t2_1, lf_scale=self.b2_1, hf_scale=self.s2_1, amplitude_scale=self.a2_1, phase_scale=self.p2_1, blend_type=self.blend2_1)

                        finetune = finetune + 1
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k1)], global_scale=self.g1, freq_threshold=self.t1, lf_scale=self.b1, hf_scale=self.s1, amplitude_scale=self.a1, phase_scale=self.p1, blend_type=self.blend1)

                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)], global_scale=self.g1_1, freq_threshold=self.t1_1, lf_scale=self.b1_1, hf_scale=self.s1_1, amplitude_scale=self.a1_1, phase_scale=self.p1_1, blend_type=self.blend1_1)

                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(hidden_states[:, :int(hidden_states.shape[1] * self.k2)], global_scale=self.g2, freq_threshold=self.t2, lf_scale=self.b2, hf_scale=self.s2, amplitude_scale=self.a2, phase_scale=self.p2, blend_type=self.blend2)

                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)], global_scale=self.g2_1, freq_threshold=self.t2_1, lf_scale=self.b2_1, hf_scale=self.s2_1, amplitude_scale=self.a2_1, phase_scale=self.p2_1, blend_type=self.blend2_1)

                        finetune = finetune + 1

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,
                        None,
                        cross_attention_kwargs,
                        attention_mask,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )[0]

                skipping = skipping + 1

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states
        
        return forward

    for i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, 'types', types)

            setattr(upsample_block, 'k1', k1)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 't1', t1)
            setattr(upsample_block, 's1', s1)

            setattr(upsample_block, 'k2', k2)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 't2', t2)
            setattr(upsample_block, 's2', s2)

            setattr(upsample_block, 'k1_1', k1_1)
            setattr(upsample_block, 'b1_1', b1_1)
            setattr(upsample_block, 't1_1', t1_1)
            setattr(upsample_block, 's1_1', s1_1)

            setattr(upsample_block, 'k2_1', k2_1)
            setattr(upsample_block, 'b2_1', b2_1)
            setattr(upsample_block, 't2_1', t2_1)
            setattr(upsample_block, 's2_1', s2_1)

            setattr(upsample_block, 'g1', g1)
            setattr(upsample_block, 'g2', g2)
            setattr(upsample_block, 'g1_1', g1_1)
            setattr(upsample_block, 'g2_1', g2_1)

            setattr(upsample_block, 'blend1', blend1)
            setattr(upsample_block, 'blend2', blend2)
            setattr(upsample_block, 'blend1_1', blend1_1)
            setattr(upsample_block, 'blend2_1', blend2_1)

            setattr(upsample_block, 'a1', a1)
            setattr(upsample_block, 'a2', a2)
            setattr(upsample_block, 'a1_1', a1_1)
            setattr(upsample_block, 'a2_1', a2_1)

            setattr(upsample_block, 'p1', p1)
            setattr(upsample_block, 'p2', p2)
            setattr(upsample_block, 'p1_1', p1_1)
            setattr(upsample_block, 'p2_1', p2_1)

            setattr(upsample_block, 'skips', skips)
            setattr(upsample_block, 'tunes', tunes)
