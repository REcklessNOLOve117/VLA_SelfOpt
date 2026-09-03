'use client';

import { useEffect, useMemo, useState } from 'react';
import NextImage from 'next/image';

type Metric = number | null;

type TaskResult = {
  task_id: number;
  task_name: string;
  trials: number;
  base_successes: number | null;
  base_sr: Metric;
  ft_successes: number | null;
  ft_sr: Metric;
  delta: Metric;
};

type Summary = {
  status: 'awaiting_data' | 'complete';
  protocol: {
    suite: string;
    tasks: number;
    states_per_task: number;
    sampling_seeds: number[];
    episodes_per_model: number;
    action_chunk: number;
    max_action_steps: number;
  };
  overall: {
    base_sr: Metric;
    ft_sr: Metric;
    delta: Metric;
    paired_cluster_bootstrap_95: [number, number] | null;
    bootstrap_samples: number;
    performance_gate: 'pending' | 'passed' | 'failed';
  };
  tasks: TaskResult[];
};

type Pair = {
  episode_key: string;
  task_id: number;
  init_state_id: number;
  sampling_seed: number;
  base: { success: boolean | null; video_path: string | null };
  ft: { success: boolean | null; video_path: string | null };
  behavior_label: string | null;
  annotation_note: string | null;
};

type PairRegistry = {
  status: 'preregistered' | 'complete';
  hero_episode_key: string | null;
  pairs: Pair[];
};

type ImaginedRollout = {
  status: 'awaiting_data' | 'ok';
  episode_key: string;
  instruction?: string | null;
  condition_frames: string[];
  actions: number[][];
  generated_frames: string[];
  rewards: number[];
};

const behaviorLabels: Record<string, string> = {
  wrong_object: '选错对象',
  reach_failure: '接近失败',
  grasp_failure: '抓取失败',
  drop: '掉落',
  placement_or_relation_failure: '放置 / 空间关系失败',
  gripper_error: '夹爪错误',
  oscillation_or_timeout: '振荡 / 超时',
  collision_or_out_of_bounds: '碰撞 / 越界',
  runtime_error: '运行错误',
};

const actionColors = ['#a8ff60', '#62d7ff', '#ffb44a', '#ec74ff', '#ff6b6b', '#7e9cff', '#f7f7f2'];
const actionNames = ['Δx', 'Δy', 'Δz', 'Δroll', 'Δpitch', 'Δyaw', 'grip'];
const pagesBasePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/$/, '');

function assetUrl(path: string) {
  if (/^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith('data:') || path.startsWith('blob:')) return path;
  if (pagesBasePath && (path === pagesBasePath || path.startsWith(`${pagesBasePath}/`))) return path;
  return `${pagesBasePath}/${path.replace(/^\/+/, '')}`;
}

function percent(value: Metric, signed = false) {
  if (value === null || !Number.isFinite(value)) return '—';
  const scaled = value * 100;
  return `${signed && scaled > 0 ? '+' : ''}${scaled.toFixed(1)}%`;
}

function outcome(value: boolean | null) {
  if (value === null) return { label: '待评测', className: 'pending' };
  return value ? { label: '成功', className: 'success' } : { label: '失败', className: 'failure' };
}

function EmptyFrame({ label }: { label: string }) {
  return <div className="empty-frame"><span>{label}</span></div>;
}

function ActionTrace({ actions }: { actions: number[][] }) {
  if (actions.length !== 8 || actions.some((row) => row.length !== 7)) {
    return <div className="chart-empty">等待 8 × 7 动作序列</div>;
  }
  const width = 640;
  const height = 164;
  const padding = 18;
  const maxAbs = Math.max(1e-6, ...actions.flat().map((value) => Math.abs(value)));
  const points = actionNames.map((_, axis) =>
    actions.map((row, step) => {
      const x = padding + step * ((width - padding * 2) / 7);
      const y = height / 2 - (row[axis] / maxAbs) * (height / 2 - padding);
      return `${x},${y}`;
    }).join(' ')
  );
  return (
    <div className="trace-wrap">
      <svg className="trace" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="8 步 7 维动作轨迹">
        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} className="zero-line" />
        {Array.from({ length: 8 }, (_, index) => {
          const x = padding + index * ((width - padding * 2) / 7);
          return <line key={index} x1={x} y1={padding} x2={x} y2={height - padding} className="step-line" />;
        })}
        {points.map((polyline, axis) => <polyline key={actionNames[axis]} points={polyline} fill="none" stroke={actionColors[axis]} strokeWidth="2" />)}
      </svg>
      <div className="trace-legend">
        {actionNames.map((name, index) => <span key={name}><i style={{ background: actionColors[index] }} />{name}</span>)}
      </div>
    </div>
  );
}

function RewardTrace({ rewards }: { rewards: number[] }) {
  if (rewards.length !== 8) return <div className="reward-empty">Reward 待生成</div>;
  const width = 400;
  const height = 96;
  const points = rewards.map((reward, index) => `${8 + index * ((width - 16) / 7)},${height - 8 - Math.max(0, Math.min(1, reward)) * (height - 16)}`).join(' ');
  return (
    <svg className="reward-trace" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="8 帧奖励曲线">
      <line x1="8" y1={height - 8} x2={width - 8} y2={height - 8} className="zero-line" />
      <polyline points={points} fill="none" stroke="#a8ff60" strokeWidth="3" />
      {rewards.map((reward, index) => <circle key={index} cx={8 + index * ((width - 16) / 7)} cy={height - 8 - Math.max(0, Math.min(1, reward)) * (height - 16)} r="3.5" fill="#a8ff60" />)}
    </svg>
  );
}

export default function ResultsDashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [registry, setRegistry] = useState<PairRegistry | null>(null);
  const [rollout, setRollout] = useState<ImaginedRollout | null>(null);
  const [selectedKey, setSelectedKey] = useState<string>('');
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(assetUrl('/results/summary.json'), { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error('summary');
        return response.json() as Promise<Summary>;
      }),
      fetch(assetUrl('/results/paired_videos.json'), { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error('pairs');
        return response.json() as Promise<PairRegistry>;
      }),
      fetch(assetUrl('/results/imagined_rollout.json'), { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error('rollout');
        return response.json() as Promise<ImaginedRollout>;
      }),
    ]).then(([summaryData, pairData, rolloutData]) => {
      if (!active) return;
      setSummary(summaryData);
      setRegistry(pairData);
      setRollout(rolloutData);
      setSelectedKey(pairData.hero_episode_key ?? pairData.pairs[0]?.episode_key ?? '');
    }).catch(() => active && setLoadError(true));
    return () => { active = false; };
  }, []);

  const selectedPair = useMemo(
    () => registry?.pairs.find((pair) => pair.episode_key === selectedKey) ?? registry?.pairs[0] ?? null,
    [registry, selectedKey],
  );
  const gate = summary?.overall.performance_gate ?? 'pending';
  const ready = summary?.status === 'complete';
  const ci = summary?.overall.paired_cluster_bootstrap_95;
  const stateLabel = loadError ? '结果文件读取失败' : ready ? (gate === 'passed' ? '显著正向 · 验收通过' : '评测完成 · 未通过正向门槛') : '等待正式评测数据';

  return (
    <main className="report-shell">
      <header className="report-header">
        <div className="wordmark"><span>WM</span><div><strong>WORLD MODEL RL</strong><small>LIBERO-SPATIAL · POC 01</small></div></div>
        <div className="protocol-tags"><span>RLinf v0.3</span><span>LoRA-only · r32</span><span>N=1,500 / model</span></div>
      </header>

      <section className="scoreboard" aria-labelledby="scoreboard-title">
        <div className="scoreboard-copy">
          <p className="kicker">PRIMARY COMPARISON</p>
          <h1 id="scoreboard-title">OpenVLA-OFT<br/><span>Base vs. World-Model GRPO</span></h1>
          <p className={`run-state ${gate}`}><i />{stateLabel}</p>
        </div>
        <div className="score-grid">
          <article className="score-card base"><small>BASE / SFT</small><strong>{percent(summary?.overall.base_sr ?? null)}</strong><span>LIBERO-Spatial 宏平均成功率</span></article>
          <article className="score-card ft"><small>FT / GRPO</small><strong>{percent(summary?.overall.ft_sr ?? null)}</strong><span>冻结 Wan + KIR · Policy LoRA</span></article>
          <article className="score-card delta"><small>PAIRED Δ</small><strong>{percent(summary?.overall.delta ?? null, true)}</strong><span>{ci ? `95% CI [${percent(ci[0], true)}, ${percent(ci[1], true)}]` : '95% CI 待计算'}</span></article>
        </div>
      </section>

      <section className="section-block task-section" aria-labelledby="tasks-title">
        <div className="section-heading">
          <div><p>01 / REAL SIMULATOR</p><h2 id="tasks-title">十项任务逐项对比</h2></div>
          <div className="legend"><span><i className="base-dot"/>Base</span><span><i className="ft-dot"/>GRPO-FT</span><span>每项 150 trials</span></div>
        </div>
        <div className="task-table" role="table" aria-label="LIBERO-Spatial 十任务成功率">
          <div className="task-row task-head" role="row"><span>任务</span><span>成功率</span><span>Base</span><span>FT</span><span>Δ</span></div>
          {(summary?.tasks ?? Array.from({ length: 10 }, (_, task_id) => ({ task_id, task_name: `Task ${task_id + 1}`, trials: 150, base_successes: null, base_sr: null, ft_successes: null, ft_sr: null, delta: null }))).map((task) => (
            <div className="task-row" role="row" key={task.task_id}>
              <span className="task-name"><b>{String(task.task_id + 1).padStart(2, '0')}</b><em>{task.task_name}</em></span>
              <span className="metric-bars">
                <i className="base-bar" style={{ width: `${(task.base_sr ?? 0) * 100}%` }} />
                <i className="ft-bar" style={{ width: `${(task.ft_sr ?? 0) * 100}%` }} />
              </span>
              <span className="number base-number">{percent(task.base_sr)}</span>
              <span className="number ft-number">{percent(task.ft_sr)}</span>
              <span className={`number delta-number ${(task.delta ?? 0) < 0 ? 'negative' : ''}`}>{percent(task.delta, true)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="section-block imagined-section" aria-labelledby="imagined-title">
        <div className="section-heading">
          <div><p>02 / WORLD MODEL</p><h2 id="imagined-title">Action-conditioned imagined rollout</h2></div>
          <div className="imagined-badge">IMAGINED · 非真值画面</div>
        </div>
        <div className="rollout-stage">
          <div className="frame-group">
            <div className="frame-group-title"><span>CONDITION</span><small>参考帧 + KIR 上下文</small></div>
            <div className="condition-strip">
              {Array.from({ length: 5 }, (_, index) => rollout?.condition_frames[index] ? <NextImage key={index} src={assetUrl(rollout.condition_frames[index])} width={256} height={256} unoptimized alt={`条件帧 ${index + 1}`} /> : <EmptyFrame key={index} label={index === 0 ? 'REF' : `K${index}`} />)}
            </div>
          </div>
          <div className="rollout-arrow" aria-hidden="true">→</div>
          <div className="frame-group generated-group">
            <div className="frame-group-title"><span>WAN · t+1 … t+8</span><small>256 × 256 · 5 inference steps</small></div>
            <div className="generated-strip">
              {Array.from({ length: 8 }, (_, index) => rollout?.generated_frames[index] ? <NextImage key={index} src={assetUrl(rollout.generated_frames[index])} width={256} height={256} unoptimized alt={`预测帧 t+${index + 1}`} /> : <EmptyFrame key={index} label={`+${index + 1}`} />)}
            </div>
          </div>
        </div>
        <div className="signal-grid">
          <article><header><span>7D ACTION TRACE</span><small>8-step chunk</small></header><ActionTrace actions={rollout?.actions ?? []}/></article>
          <article><header><span>RESNET REWARD</span><small>success probability</small></header><RewardTrace rewards={rollout?.rewards ?? []}/></article>
        </div>
      </section>

      <section className="section-block video-section" aria-labelledby="video-title">
        <div className="section-heading video-heading">
          <div><p>03 / PAIRED EXECUTION</p><h2 id="video-title">相同初始状态与 sampling seed</h2></div>
          <label className="episode-select">对比样本
            <select value={selectedPair?.episode_key ?? ''} onChange={(event) => setSelectedKey(event.target.value)}>
              {(registry?.pairs ?? []).map((pair) => <option key={pair.episode_key} value={pair.episode_key}>{`Task ${pair.task_id + 1} · State ${pair.init_state_id} · Seed ${pair.sampling_seed}`}</option>)}
            </select>
          </label>
        </div>
        {selectedPair ? (
          <>
            <div className="pair-meta">
              <code>{selectedPair.episode_key}</code>
              <span>{selectedPair.behavior_label ? behaviorLabels[selectedPair.behavior_label] ?? selectedPair.behavior_label : '行为标签待标注'}</span>
              {selectedPair.annotation_note && <p>{selectedPair.annotation_note}</p>}
            </div>
            <div className="video-grid">
              {(['base', 'ft'] as const).map((model) => {
                const result = selectedPair[model];
                const resultLabel = outcome(result.success);
                return (
                  <article className={`video-card ${model}`} key={model}>
                    <header><div><small>{model === 'base' ? 'BASE / SFT' : 'FT / GRPO'}</small><strong>{model === 'base' ? 'OpenVLA-OFT' : 'World-Model Fine-tuned'}</strong></div><span className={resultLabel.className}>{resultLabel.label}</span></header>
                    {result.video_path ? <video key={result.video_path} controls muted playsInline preload="metadata"><source src={assetUrl(result.video_path)} type="video/mp4"/></video> : <div className="video-empty"><span>{model === 'base' ? 'BASE' : 'FT'}</span><p>视频将在固定 episode 完成后出现</p></div>}
                  </article>
                );
              })}
            </div>
            <div className="pair-index" aria-label="20 组预注册对比">
              {(registry?.pairs ?? []).map((pair, index) => <button key={pair.episode_key} className={pair.episode_key === selectedPair.episode_key ? 'active' : ''} onClick={() => setSelectedKey(pair.episode_key)} aria-label={`选择第 ${index + 1} 组对比`}>{String(index + 1).padStart(2, '0')}</button>)}
            </div>
          </>
        ) : <div className="video-empty standalone"><span>20 PAIRS</span><p>预注册清单加载中</p></div>}
      </section>

      <footer>
        <p>训练 rollout：冻结 action-conditioned Wan 世界模型 · KIR · ResNet RM · GRPO</p>
        <p>真值评测：LIBERO simulator · paired cluster bootstrap × 10,000</p>
      </footer>
    </main>
  );
}
