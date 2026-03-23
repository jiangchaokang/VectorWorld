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
  registerVideo: (node: HTMLVideoElement | null) => void;
  onLoadedMetadata: (video: HTMLVideoElement) => void;
};

const METHOD_ORDER: MethodKey[] = ["scenedream", "fewstep", "onestep"];
const PLAYBACK_FPS = 24;

const activePillButtonClass =
  "rounded-full bg-zinc-900 px-4 py-2 text-sm font-medium text-white dark:bg-zinc-100 dark:text-zinc-900";

const inactivePillButtonClass =
  "rounded-full border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800";

const primaryButtonClass =
  "rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-400";

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

function toneBadgeClasses(accent: AccentTone) {
  switch (accent) {
    case "blue":
      return "bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-300";
    case "violet":
      return "bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300";
    default:
      return "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300";
  }
}

function toneCardClasses(accent: AccentTone) {
  switch (accent) {
    case "blue":
      return "border-blue-200 dark:border-blue-900";
    case "violet":
      return "border-violet-200 dark:border-violet-900";
    default:
      return "border-zinc-200 dark:border-zinc-800";
  }
}

function MethodCard({
  method,
  src,
  metrics,
  currentTime,
  registerVideo,
  onLoadedMetadata,
}: MethodCardProps) {
  const aspectRatio =
    metrics?.aspectRatio && metrics.aspectRatio > 0 ? metrics.aspectRatio : 1;

  return (
    <article
      className={`rounded-3xl border bg-white p-4 shadow-sm dark:bg-zinc-900 ${toneCardClasses(method.accent)}`}
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="space-y-1.5">
          <div
            className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${toneBadgeClasses(method.accent)}`}
          >
            {method.badge}
          </div>

          <h3 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            {method.label}
          </h3>

          <p className="text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            {method.note}
          </p>
        </div>

        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
        >
          Open
        </a>
      </div>

      <div
        className="overflow-hidden rounded-2xl border border-zinc-200 bg-black shadow-inner dark:border-zinc-800"
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

      <div className="mt-3 flex flex-wrap gap-2 text-xs text-zinc-600 dark:text-zinc-400">
        <span className="rounded-full bg-zinc-100 px-3 py-1 dark:bg-zinc-950">
          clip {formatTime(metrics?.duration)}
        </span>
        <span className="rounded-full bg-zinc-100 px-3 py-1 dark:bg-zinc-950">
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
      <section className="rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="space-y-4">
          <div>
            <p className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">Representative clips</p>
            <div className="flex flex-wrap gap-2">
              {cases.map((item, idx) => (
                <button key={item.key} type="button" onClick={() => setCaseIndex(idx)}
                  className={idx === caseIndex ? activePillButtonClass : inactivePillButtonClass}>
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* ===== prev / next navigation ===== */}
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={goPrev} className={inactivePillButtonClass}>← Previous</button>
            <button type="button" onClick={goNext} className={activePillButtonClass}>Next →</button>
            <span className="rounded-full border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300">
              {selectedCase.label} · {caseIndex + 1} / {cases.length}
            </span>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{selectedCase.label}</p>
                  <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">matched rollout progress</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{selectedCase.description}</p>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">Playback speed</p>
                <div className="flex flex-wrap gap-2">
                  {[0.75, 1, 1.5].map((value) => (
                    <button key={value} type="button" onClick={() => setPlaybackRate(value)}
                      className={playbackRate === value ? activePillButtonClass : inactivePillButtonClass}>
                      {value.toFixed(2).replace(".00", "")}×
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between text-sm font-medium text-zinc-700 dark:text-zinc-300">
                <span>Shared progress</span>
                <span>{Math.round(progress * 100)}%</span>
              </div>
              <input type="range" min={0} max={1000} step={1} value={Math.round(progress * 1000)}
                onChange={(e) => handleSliderChange(Number(e.target.value) / 1000)} className="w-full accent-blue-600" />
              <div className="mt-4 flex flex-wrap gap-2">
                <button type="button" onClick={togglePlayback} className={primaryButtonClass}>{isPlaying ? "Pause" : "Play"}</button>
                <button type="button" onClick={() => jumpBy(-0.05)} className={inactivePillButtonClass}>−5%</button>
                <button type="button" onClick={() => jumpBy(0.05)} className={inactivePillButtonClass}>+5%</button>
                <button type="button" onClick={resetProgress} className={inactivePillButtonClass}>Reset</button>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                <span className="rounded-full bg-white px-3 py-1 dark:bg-zinc-900">shared progress control</span>
                <span className="rounded-full bg-white px-3 py-1 dark:bg-zinc-900">reference clip {formatTime(masterDuration)}</span>
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
        <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">Generator</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">One-step MeanFlow + JVP</p>
          <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">Solver-free masked completion for repeated rollout-time generation.</p>
        </div>
        <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">Deployment cost</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">5.6 ms / 64 m × 64 m tile</p>
          <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">The online operating point reported in the paper.</p>
        </div>
        <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">Step budget</p>
          <p className="mt-2 text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">3–5 steps</p>
          <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">Higher fidelity when a small offline budget.</p>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.86fr)]">
        <section className="rounded-3xl border border-zinc-200 bg-zinc-50 p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">What to inspect</h3>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            <li>Lane continuity at the frontier.</li>
            <li>Route continuation under a tight step budget.</li>
            <li>Agent-map consistency during low-latency generation.</li>
          </ul>
        </section>
        <section className="rounded-3xl border border-zinc-200 bg-zinc-50 p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Reading guide</h3>
          <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            One-step MeanFlow is the deployment point. Few-step flow recovers fidelity when a small extra budget is allowed. The multi-step baseline remains visibly slower.
          </p>
        </section>
      </div>
    </div>
  );
}