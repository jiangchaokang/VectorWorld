import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type MethodKey = "scenedream" | "fewstep" | "onestep";
export type AccentTone = "neutral" | "blue" | "violet";

export type MethodMeta = {
  key: MethodKey;
  label: string;
  badge: string;
  note: string;
  accent: AccentTone;
};

export type CaseConfig = {
  key: string;
  label: string;
  description: string;
  videos: Record<MethodKey, string>;
};

type Props = {
  methods: Record<MethodKey, MethodMeta>;
  cases: CaseConfig[];
};

type VideoMetrics = {
  duration?: number;
  aspectRatio?: number;
};

type MetricsMap = Partial<Record<MethodKey, VideoMetrics>>;

type MethodCardProps = {
  method: MethodMeta;
  src: string;
  metrics?: VideoMetrics;
  currentTime?: number;
  isSelected: boolean;
  onSelect: () => void;
  registerVideo: (node: HTMLVideoElement | null) => void;
  onLoadedMetadata: (video: HTMLVideoElement) => void;
};

const METHOD_ORDER: MethodKey[] = ["scenedream", "fewstep", "onestep"];
const PLAYBACK_FPS = 24;

const SHORT_LABEL: Record<MethodKey, string> = {
  scenedream: "Multi-step",
  fewstep: "Few-step",
  onestep: "One-step",
};

// Latency anchored to the one reported number (5.6 ms one-step). Few-step and
// multi-step are representative, derived from solver-step count.
const LATENCY: Record<MethodKey, { ms: number; label: string; steps: string }> = {
  scenedream: { ms: 150, label: "≫ 100 ms", steps: "many solver steps" },
  fewstep: { ms: 22, label: "≈ 17–28 ms", steps: "3–5 steps" },
  onestep: { ms: 5.6, label: "5.6 ms", steps: "1 step" },
};

const AXIS_MIN = 3.5;
const AXIS_MAX = 260;
const BUDGET_MS = 10;

function axisPct(ms: number) {
  const t =
    (Math.log(ms) - Math.log(AXIS_MIN)) /
    (Math.log(AXIS_MAX) - Math.log(AXIS_MIN));
  return Math.max(0, Math.min(1, t)) * 100;
}

const selectedPillClass =
  "rounded-full border border-[color:var(--vw-accent-line)] bg-brand-soft px-4 py-2 text-sm font-medium text-brand transition";

const inactivePillClass =
  "rounded-full border border-hairline-strong px-4 py-2 text-sm font-medium text-ink-2 transition hover:border-[color:var(--vw-accent-line)] hover:text-ink-1 active:scale-[0.98]";

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatTime(value?: number) {
  if (!isFiniteNumber(value)) return "—";
  return `${value.toFixed(2)} s`;
}

function LatencyAxis({
  methods,
  selected,
  onSelect,
}: {
  methods: Record<MethodKey, MethodMeta>;
  selected: MethodKey;
  onSelect: (key: MethodKey) => void;
}) {
  const budgetPct = axisPct(BUDGET_MS);

  return (
    <section className="rounded-3xl border border-hairline bg-surface-1 p-5 md:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <span className="vw-eyebrow">Latency axis</span>
          <h3 className="mt-2 text-sm font-semibold text-ink-1">
            Pick an operating point
          </h3>
        </div>
        <span className="rounded-full bg-surface-2 px-3 py-1 font-mono text-[0.7rem] font-medium text-ink-2">
          log scale · per 64 m tile
        </span>
      </div>

      <div className="relative mx-2 mt-12 mb-12 h-px bg-[color:var(--vw-hairline-strong)] md:mx-4">
        <div
          className="absolute inset-y-[-16px] left-0 rounded-l-xl border-r border-dashed border-[color:var(--vw-accent-line)] bg-brand-soft"
          style={{ width: `${budgetPct}%` }}
          aria-hidden="true"
        />
        <span
          className="absolute -top-9 whitespace-nowrap font-mono text-[0.62rem] font-medium uppercase tracking-wider text-brand"
          style={{ left: `${budgetPct}%`, transform: "translateX(-50%)" }}
        >
          ◂ streaming budget
        </span>

        {METHOD_ORDER.map((key) => {
          const lat = LATENCY[key];
          const pct = axisPct(lat.ms);
          const isSel = selected === key;
          return (
            <button
              key={key}
              type="button"
              onClick={() => onSelect(key)}
              className="group absolute top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center"
              style={{ left: `${pct}%` }}
              aria-pressed={isSel}
              aria-label={`${methods[key].label}, ${lat.label}, ${lat.steps}`}
            >
              <span
                className={`absolute -top-9 whitespace-nowrap text-[0.72rem] font-medium transition ${
                  isSel ? "text-ink-1" : "text-ink-2 group-hover:text-ink-1"
                }`}
              >
                {SHORT_LABEL[key]}
              </span>
              <span
                className={`h-3.5 w-3.5 rounded-full border-2 transition ${
                  isSel
                    ? "border-[color:var(--vw-accent)] bg-[color:var(--vw-accent)] shadow-[0_0_0_5px_var(--vw-accent-soft)]"
                    : "border-[color:var(--vw-hairline-strong)] bg-[color:var(--vw-surface-1)] group-hover:border-[color:var(--vw-accent-line)]"
                }`}
              />
              <span
                className={`absolute top-6 whitespace-nowrap font-mono text-[0.7rem] transition ${
                  isSel ? "text-brand" : "text-ink-3"
                }`}
              >
                {lat.label}
              </span>
            </button>
          );
        })}
      </div>

      <p className="mt-2 text-xs leading-5 text-ink-3">
        Latency anchored to the reported{" "}
        <span className="font-mono text-ink-2">5.6 ms</span> one-step cost;
        few-step scales with 3–5 solver steps, and the multi-step baseline needs
        many more. Only the one-step point sits inside the streaming budget.
      </p>
    </section>
  );
}

function MethodCard({
  method,
  src,
  metrics,
  currentTime,
  isSelected,
  onSelect,
  registerVideo,
  onLoadedMetadata,
}: MethodCardProps) {
  const aspectRatio =
    metrics?.aspectRatio && metrics.aspectRatio > 0 ? metrics.aspectRatio : 1;

  return (
    <article
      className={`vw-card overflow-hidden p-4 transition ${
        isSelected
          ? "border-[color:var(--vw-accent-line)] shadow-[0_0_0_1px_var(--vw-accent-line)]"
          : ""
      }`}
      aria-current={isSelected ? "true" : undefined}
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="space-y-1.5">
          <button
            type="button"
            onClick={onSelect}
            className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold transition ${
              isSelected
                ? "bg-brand text-brand-contrast"
                : "bg-surface-2 text-ink-2 hover:text-ink-1"
            }`}
            aria-pressed={isSelected}
          >
            {method.badge}
          </button>

          <h3 className="text-lg font-semibold tracking-tight text-ink-1">
            {method.label}
          </h3>

          <p className="text-sm leading-6 text-ink-2">{method.note}</p>
        </div>

        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="vw-pill !px-3 !py-1 !text-xs"
        >
          Open
        </a>
      </div>

      <div
        className="overflow-hidden rounded-2xl border border-hairline bg-[var(--vw-canvas)]"
        style={{ aspectRatio }}
      >
        <video
          ref={registerVideo}
          src={src}
          className="block h-full w-full object-contain"
          muted
          playsInline
          controls={false}
          preload="metadata"
          onLoadedMetadata={(event) => onLoadedMetadata(event.currentTarget)}
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-2 font-mono text-xs text-ink-3">
        <span className="rounded-full bg-surface-2 px-3 py-1">
          clip {formatTime(metrics?.duration)}
        </span>
        <span className="rounded-full bg-surface-2 px-3 py-1">
          current {formatTime(currentTime)}
        </span>
      </div>
    </article>
  );
}

export default function EfficiencyExplorerClient({ methods, cases }: Props) {
  const [caseIndex, setCaseIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [metrics, setMetrics] = useState<MetricsMap>({});
  const [selectedMethod, setSelectedMethod] = useState<MethodKey>("onestep");

  const progressRef = useRef(progress);
  const intervalRef = useRef<number | null>(null);
  const lastTickRef = useRef<number | null>(null);

  const videoRefs = useRef<Record<MethodKey, HTMLVideoElement | null>>({
    scenedream: null,
    fewstep: null,
    onestep: null,
  });

  const selectedCase = cases[caseIndex] ?? cases[0] ?? null;
  const caseKey = selectedCase?.key ?? "";

  const masterDuration = useMemo(() => {
    const values = METHOD_ORDER.map((key) => metrics[key]?.duration).filter(
      (value): value is number => isFiniteNumber(value) && value > 0,
    );
    return values.length > 0 ? Math.max(...values) : undefined;
  }, [metrics]);

  const syncVideosToProgress = useCallback((nextProgress: number) => {
    const clamped = clamp01(nextProgress);
    METHOD_ORDER.forEach((key) => {
      const video = videoRefs.current[key];
      if (!video) return;
      if (!Number.isFinite(video.duration) || video.duration <= 0) return;
      const targetTime = Math.min(clamped * video.duration, Math.max(video.duration - 0.001, 0));
      if (Math.abs(video.currentTime - targetTime) > 0.015) {
        try { video.currentTime = targetTime; } catch { /* ignore */ }
      }
      video.pause();
    });
  }, []);

  useEffect(() => {
    progressRef.current = progress;
    syncVideosToProgress(progress);
  }, [progress, caseKey, syncVideosToProgress]);

  useEffect(() => {
    setProgress(0);
    setIsPlaying(false);
    setMetrics({});
  }, [caseIndex]);

  useEffect(() => {
    if (!isPlaying) {
      if (intervalRef.current !== null) { window.clearInterval(intervalRef.current); intervalRef.current = null; }
      lastTickRef.current = null;
      return;
    }
    const refDur = masterDuration ?? 5;
    lastTickRef.current = performance.now();
    intervalRef.current = window.setInterval(() => {
      const now = performance.now();
      const prev = lastTickRef.current ?? now;
      const delta = (now - prev) / 1000;
      lastTickRef.current = now;
      let shouldStop = false;
      setProgress((c) => { const n = c + (delta * playbackRate) / refDur; if (n >= 1) { shouldStop = true; return 1; } return n; });
      if (shouldStop) setIsPlaying(false);
    }, 1000 / PLAYBACK_FPS);
    return () => { if (intervalRef.current !== null) { window.clearInterval(intervalRef.current); intervalRef.current = null; } lastTickRef.current = null; };
  }, [isPlaying, playbackRate, masterDuration, caseKey]);

  if (!selectedCase) return null;

  const handleSliderChange = (value: number) => { setIsPlaying(false); setProgress(clamp01(value)); };
  const jumpBy = (delta: number) => { setIsPlaying(false); setProgress((c) => clamp01(c + delta)); };
  const resetProgress = () => { setIsPlaying(false); setProgress(0); };
  const togglePlayback = () => { if (progress >= 1) setProgress(0); setIsPlaying((c) => !c); };

  const goPrev = () => setCaseIndex((c) => (c === 0 ? cases.length - 1 : c - 1));
  const goNext = () => setCaseIndex((c) => (c === cases.length - 1 ? 0 : c + 1));

  return (
    <div className="space-y-6">
      <LatencyAxis methods={methods} selected={selectedMethod} onSelect={setSelectedMethod} />

      <section className="rounded-3xl border border-hairline bg-surface-1 p-5">
        <div className="space-y-4">
          <div>
            <p className="mb-3 text-sm font-medium text-ink-2">Representative clips</p>
            <div className="flex flex-wrap gap-2">
              {cases.map((item, idx) => (
                <button key={item.key} type="button" onClick={() => setCaseIndex(idx)}
                  className={idx === caseIndex ? selectedPillClass : inactivePillClass}>
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* ===== prev / next navigation ===== */}
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={goPrev} className={inactivePillClass}>← Previous</button>
            <button type="button" onClick={goNext} className={inactivePillClass}>Next →</button>
            <span className="rounded-full border border-hairline-strong bg-surface-1 px-4 py-2 text-sm font-medium text-ink-2">
              {selectedCase.label} · {caseIndex + 1} / {cases.length}
            </span>
          </div>

          <div className="rounded-2xl border border-hairline bg-surface-2 p-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-ink-1">{selectedCase.label}</p>
                  <span className="rounded-full bg-surface-1 px-3 py-1 text-xs font-medium text-ink-2">matched rollout progress</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-ink-2">{selectedCase.description}</p>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium text-ink-2">Playback speed</p>
                <div className="flex flex-wrap gap-2">
                  {[0.75, 1, 1.5].map((value) => (
                    <button key={value} type="button" onClick={() => setPlaybackRate(value)}
                      className={playbackRate === value ? selectedPillClass : inactivePillClass}>
                      {value.toFixed(2).replace(".00", "")}×
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between text-sm font-medium text-ink-2">
                <span>Shared progress</span>
                <span className="font-mono text-ink-1">{Math.round(progress * 100)}%</span>
              </div>
              <input type="range" min={0} max={1000} step={1} value={Math.round(progress * 1000)}
                onChange={(e) => handleSliderChange(Number(e.target.value) / 1000)}
                className="w-full [accent-color:var(--vw-accent)]" />
              <div className="mt-4 flex flex-wrap gap-2">
                <button type="button" onClick={togglePlayback} className="vw-button-primary">{isPlaying ? "Pause" : "Play"}</button>
                <button type="button" onClick={() => jumpBy(-0.05)} className={inactivePillClass}>−5%</button>
                <button type="button" onClick={() => jumpBy(0.05)} className={inactivePillClass}>+5%</button>
                <button type="button" onClick={resetProgress} className={inactivePillClass}>Reset</button>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 font-mono text-xs text-ink-3">
                <span className="rounded-full bg-surface-1 px-3 py-1">shared progress control</span>
                <span className="rounded-full bg-surface-1 px-3 py-1">reference clip {formatTime(masterDuration)}</span>
              </div>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            {METHOD_ORDER.map((key) => {
              const method = methods[key];
              const src = selectedCase.videos[key];
              const mMet = metrics[key];
              const dur = mMet?.duration;
              const curTime = isFiniteNumber(dur) && dur > 0 ? dur * progress : undefined;
              return (
                <MethodCard
                  key={`${selectedCase.key}-${key}`}
                  method={method}
                  src={src}
                  metrics={mMet}
                  currentTime={curTime}
                  isSelected={selectedMethod === key}
                  onSelect={() => setSelectedMethod(key)}
                  registerVideo={(node) => { videoRefs.current[key] = node; }}
                  onLoadedMetadata={(video) => {
                    const ar = video.videoWidth > 0 && video.videoHeight > 0 ? video.videoWidth / video.videoHeight : undefined;
                    setMetrics((c) => ({
                      ...c,
                      [key]: {
                        duration: Number.isFinite(video.duration) ? video.duration : c[key]?.duration,
                        aspectRatio: ar ?? c[key]?.aspectRatio,
                      },
                    }));
                    syncVideosToProgress(progressRef.current);
                  }}
                />
              );
            })}
          </div>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-3xl border border-hairline bg-surface-2 p-4">
          <p className="vw-stat-label">Generator</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-ink-1">One-step MeanFlow + JVP</p>
          <p className="mt-2 text-sm leading-6 text-ink-2">Solver-free masked completion for repeated rollout-time generation.</p>
        </div>
        <div className="rounded-3xl border border-hairline bg-surface-2 p-4">
          <p className="vw-stat-label">Deployment cost</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-ink-1"><span className="vw-accent-text">5.6 ms</span> / 64 m × 64 m tile</p>
          <p className="mt-2 text-sm leading-6 text-ink-2">The online operating point reported in the paper.</p>
        </div>
        <div className="rounded-3xl border border-hairline bg-surface-2 p-4">
          <p className="vw-stat-label">Step budget</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-ink-1">3–5 steps</p>
          <p className="mt-2 text-sm leading-6 text-ink-2">Higher fidelity when a small offline budget.</p>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.86fr)]">
        <section className="rounded-3xl border border-hairline bg-surface-2 p-5">
          <h3 className="text-sm font-semibold text-ink-1">What to inspect</h3>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-ink-2">
            <li>Lane continuity at the frontier.</li>
            <li>Route continuation under a tight step budget.</li>
            <li>Agent-map consistency during low-latency generation.</li>
          </ul>
        </section>
        <section className="rounded-3xl border border-hairline bg-surface-2 p-5">
          <h3 className="text-sm font-semibold text-ink-1">Reading guide</h3>
          <p className="mt-3 text-sm leading-6 text-ink-2">
            One-step MeanFlow is the deployment point. Few-step flow recovers fidelity when a small extra budget is allowed. The multi-step baseline remains visibly slower.
          </p>
        </section>
      </div>
    </div>
  );
}
