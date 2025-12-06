'use client';

import { isVideoModelFromArch, groupedModelOptions } from './options';
import { defaultDatasetConfig } from './jobConfig';
import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { TextInput, SelectInput, Checkbox, FormGroup, NumberInput } from '@/components/formInputs';
import Card from '@/components/Card';
import { X, FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
};

const isDev = process.env.NODE_ENV === 'development';

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
}: Props) {
  const isVideoModel = isVideoModelFromArch(jobConfig.config.process[0].model.arch);
  return (
    <>
      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          <Card title="Job">
            <TextInput
              label="Training Name"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="Enter training name"
              disabled={runId !== null}
              required
            />
            <SelectInput
              label="GPU ID"
              value={`${gpuIDs}`}
              docKey="gpuids"
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
            <TextInput
              label="Trigger Word"
              value={jobConfig.config.process[0].trigger_word || ''}
              docKey="config.process[0].trigger_word"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].trigger_word');
              }}
              placeholder=""
              required
            />
          </Card>

          {/* Model Configuration Section */}
          <Card title="Model">
            <SelectInput
              label="Model Architecture"
              value={jobConfig.config.process[0].model.arch}
              onChange={value => {
                handleModelArchChange(jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
              }}
              options={groupedModelOptions}
            />
            <TextInput
              label="Name or Path"
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={value => setJobConfig(value, 'config.process[0].model.name_or_path')}
              placeholder="e.g. ostris/Flex.1-alpha"
            />
          </Card>

          {/* Quantization Section */}
          <Card title="Quantization">
            <SelectInput
              label="Transformer"
              value={jobConfig.config.process[0].model.quantize ? 'qfloat8' : 'none'}
              onChange={value => setJobConfig(value !== 'none', 'config.process[0].model.quantize')}
              options={[
                { value: 'qfloat8', label: 'float8 (default)' },
                { value: 'none', label: 'None (Full Precision)' },
              ]}
            />
            <SelectInput
              label="Text Encoder"
              value={jobConfig.config.process[0].model.quantize_te ? 'qfloat8' : 'none'}
              onChange={value => setJobConfig(value !== 'none', 'config.process[0].model.quantize_te')}
              options={[
                { value: 'qfloat8', label: 'float8 (default)' },
                { value: 'none', label: 'None (Full Precision)' },
              ]}
            />
          </Card>
          <Card title="Target">
            <SelectInput
              label="Target Type"
              value={jobConfig.config.process[0].network?.type ?? 'lora'}
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={[
                { value: 'lora', label: 'LoRA' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {jobConfig.config.process[0].network?.type == 'lokr' && (
              <SelectInput
                label="LoKr Factor"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <NumberInput
                label="Linear Rank"
                value={jobConfig.config.process[0].network.linear}
                onChange={value => {
                  console.log('onChange', value);
                  setJobConfig(value, 'config.process[0].network.linear');
                  setJobConfig(value, 'config.process[0].network.linear_alpha');
                }}
                placeholder="eg. 16"
                min={0}
                max={1024}
                required
              />
            )}
          </Card>
          <Card title="Save">
            <SelectInput
              label="Data Type"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="Save Every"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="Max Step Saves to Keep"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
          </Card>
        </div>
        <div>
          <Card title="Training">
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
              <div>
                <NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                <NumberInput
                  label="Steps"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                />
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'adafactor', label: 'Adafactor' },
                  ]}
                />
                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
              </div>
              <div>
                <SelectInput
                  label="Timestep Type"
                  value={jobConfig.config.process[0].train.timestep_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                  options={[
                    { value: 'sigmoid', label: 'Sigmoid' },
                    { value: 'linear', label: 'Linear' },
                    { value: 'shift', label: 'Shift' },
                    { value: 'weighted', label: 'Weighted' },
                  ]}
                />
                <SelectInput
                  label="Timestep Bias"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'High Noise' },
                    { value: 'style', label: 'Low Noise' },
                  ]}
                />
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                  ]}
                />
              </div>
              <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}
                <FormGroup label="Text Encoder Optimizations" className="pt-2">
                  <Checkbox
                    label="Unload TE"
                    checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                    docKey="train.unload_text_encoder"
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                      }
                    }}
                  />
                  <Checkbox
                    label="Cache Text Embeddings"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey="train.cache_text_embeddings"
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                </FormGroup>
              </div>
              <div>
                <FormGroup label="Regularization">
                  <></>
                </FormGroup>
                <Checkbox
                  label="Differential Output Preservation"
                  docKey="train.diff_output_preservation"
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                    if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                      setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.diff_output_preservation && (
                  <>
                    <NumberInput
                      label="DOP Loss Multiplier"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                      onChange={value => setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')}
                      placeholder="eg. 1.0"
                      min={0}
                    />
                    <TextInput
                      label="DOP Preservation Class"
                      className="pt-2 pb-4"
                      value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                      onChange={value => setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')}
                      placeholder="eg. woman"
                    />
                  </>
                )}
                <Checkbox
                  label="Blank Prompt Preservation"
                  docKey="train.blank_prompt_preservation"
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                    if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                      setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.blank_prompt_preservation && (
                  <NumberInput
                    label="BPP Loss Multiplier"
                    className="pt-2"
                    value={(jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')}
                    placeholder="eg. 1.0"
                    min={0}
                  />
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="Advanced" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="Do Differential Guidance"
                  docKey="train.do_differential_guidance"
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="Differential Guidance Scale"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="Datasets">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                  <button
                    type="button"
                    onClick={() =>
                      setJobConfig(
                        jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                        'config.process[0].datasets',
                      )
                    }
                    className="absolute top-2 right-2 bg-red-800 hover:bg-red-700 rounded-full p-1 text-sm transition-colors"
                  >
                    <X />
                  </button>
                  <h2 className="text-lg font-bold mb-4">Dataset {i + 1}</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div>
                      <SelectInput
                        label="Target Dataset"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={datasetOptions}
                      />
                      <NumberInput
                        label="LoRA Weight"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                    </div>
                    <div>
                      <TextInput
                        label="Default Caption"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <NumberInput
                        label="Caption Dropout Rate"
                        className="pt-2"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                    </div>
                    <div>
                      <FormGroup label="Settings" className="">
                        <Checkbox
                          label="Cache Latents"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                          }
                        />
                        <Checkbox
                          label="Is Regularization"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                      </FormGroup>
                      <FormGroup label="Flipping" docKey="datasets.flip" className="mt-2">
                        <Checkbox
                          label={
                            <>
                              Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_x || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                        />
                        <Checkbox
                          label={
                            <>
                              Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_y || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                        />
                      </FormGroup>
                    </div>
                    <div>
                      <FormGroup label="Resolutions" className="pt-2">
                        <div className="grid grid-cols-2 gap-2">
                          {[
                            [256, 512, 768],
                            [1024, 1280, 1536],
                          ].map(resGroup => (
                            <div key={resGroup[0]} className="space-y-2">
                              {resGroup.map(res => (
                                <Checkbox
                                  key={res}
                                  label={res.toString()}
                                  checked={dataset.resolution.includes(res)}
                                  onChange={value => {
                                    const resolutions = dataset.resolution.includes(res)
                                      ? dataset.resolution.filter(r => r !== res)
                                      : [...dataset.resolution, res];
                                    setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                  }}
                                />
                              ))}
                            </div>
                          ))}
                        </div>
                      </FormGroup>
                    </div>
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() =>
                  setJobConfig(
                    [...jobConfig.config.process[0].datasets, objectCopy(defaultDatasetConfig)],
                    'config.process[0].datasets',
                  )
                }
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Add Dataset
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="Sample">
            <div
              className={
                isVideoModel
                  ? 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6'
                  : 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6'
              }
            >
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <SelectInput
                  label="Sampler"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={[
                    { value: 'flowmatch', label: 'FlowMatch' },
                    { value: 'ddpm', label: 'DDPM' },
                  ]}
                />
                <NumberInput
                  label="Guidance Scale"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="Sample Steps"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>
              <div>
                <NumberInput
                  label="Width"
                  value={jobConfig.config.process[0].sample.width}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                  placeholder="eg. 1024"
                  min={0}
                  required
                />
                <NumberInput
                  label="Height"
                  value={jobConfig.config.process[0].sample.height}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                  placeholder="eg. 1024"
                  className="pt-2"
                  min={0}
                  required
                />
                {isVideoModel && (
                  <>
                    <NumberInput
                      label="Num Frames"
                      value={jobConfig.config.process[0].sample.num_frames}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                      placeholder="eg. 0"
                      className="pt-2"
                      min={0}
                      required
                    />
                    <NumberInput
                      label="FPS"
                      value={jobConfig.config.process[0].sample.fps}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                      placeholder="eg. 0"
                      className="pt-2"
                      min={0}
                      required
                    />
                  </>
                )}
              </div>
              <div>
                <NumberInput
                  label="Seed"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="Walk Seed"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey="train.force_first_sample"
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            <FormGroup label={`Sample Prompts (${jobConfig.config.process[0].sample.samples.length})`} className="pt-2">
              <div></div>
            </FormGroup>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        <TextInput
                          label="Prompt"
                          value={sample.prompt}
                          onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                          placeholder="Enter prompt"
                          required
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
                          <TextInput
                            label="Width"
                            value={sample.width ? `${sample.width}` : ''}
                            onChange={value => {
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].width;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                          />
                          <TextInput
                            label="Height"
                            value={sample.height ? `${sample.height}` : ''}
                            onChange={value => {
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].height;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                          />
                          <TextInput
                            label="Seed"
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          <TextInput
                            label="LoRA Scale"
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                              }
                            }}
                            placeholder="1.0 (default)"
                          />
                        </div>
                      </div>
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig([...jobConfig.config.process[0].sample.samples, { prompt: '' }], 'config.process[0].sample.samples')
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              Add Prompt
            </button>
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
    </>
  );
}
