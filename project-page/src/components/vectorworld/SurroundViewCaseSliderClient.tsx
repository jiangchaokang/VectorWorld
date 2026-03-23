import { useState } from "react";

export type SurroundViewCase = {
  key: string;
  label: string;
  summary: string;
  vectorSrc: string;
  layoutSrc: string;
  resultSrc: string;
};

type Props = {
  cases: SurroundViewCase[];
};

type Tone = "zinc" | "blue" | "violet";

type StageCardProps = {
  title: string;
  badge: string;
  caption: string;
  src: string;
  tone: Tone;
  colSpan: string;
};

const activePillButtonClass =
  "rounded-full bg-zinc-900 px-4 py-2 text-sm font-medium text-white dark:bg-zinc-100 dark:text-zinc-900";

const inactivePillButtonClass =
  "rounded-full border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800";

function sectionToneClass(tone: Tone) {
  switch (tone) {
    case "blue":
      return "border-blue-200 bg-blue-50/70 dark:border-blue-500/20 dark:bg-blue-500/5";
    case "violet":
      return "border-violet-200 bg-violet-50/70 dark:border-violet-500/20 dark:bg-violet-500/5";
    default:
      return "border-zinc-200 bg-zinc-50/90 dark:border-zinc-800 dark:bg-zinc-950/70";
  }
}

function badgeToneClass(tone: Tone) {
  switch (tone) {
    case "blue":
      return "bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-300";
    case "violet":
      return "bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300";
    default:
      return "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300";
  }
}

function StageCard({ title, badge, caption, src, tone, colSpan }: StageCardProps) {
  return (
    <section className={`${colSpan} rounded-2xl border p-4 shadow-sm ${sectionToneClass(tone)}`}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</p>
        <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${badgeToneClass(tone)}`}>{badge}</span>
      </div>
      <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-950 shadow-inner dark:border-zinc-800">
        <div className="flex h-52 items-center justify-center md:h-60 xl:h-72">
          <video src={src} controls muted playsInline preload="metadata" className="h-full w-full object-contain" />
        </div>
      </div>
      <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{caption}</p>
    </section>
  );
}

export default function SurroundViewCaseSliderClient({ cases }: Props) {
  const [index, setIndex] = useState(0);
  if (cases.length === 0) return null;
  const current = cases[index] ?? cases[0];

  const goPrev = () => setIndex((c) => (c === 0 ? cases.length - 1 : c - 1));
  const goNext = () => setIndex((c) => (c === cases.length - 1 ? 0 : c + 1));

  return (
    <div className="space-y-5">
      <section className="rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="space-y-3">
            <div>
              <p className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">Cases</p>
              <div className="flex flex-wrap gap-2">
                {cases.map((item, i) => (
                  <button key={item.key} type="button" onClick={() => setIndex(i)}
                    className={i === index ? activePillButtonClass : inactivePillButtonClass}>
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={goPrev} className={inactivePillButtonClass}>← Previous</button>
              <button type="button" onClick={goNext} className={activePillButtonClass}>Next →</button>
              <span className="rounded-full border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300">
                {current.label} · {index + 1} / {cases.length}
              </span>
            </div>
          </div>

          <div className="max-w-3xl rounded-2xl border border-violet-200 bg-violet-50/70 p-4 dark:border-violet-500/20 dark:bg-violet-500/5">
            <p className="text-xs font-semibold uppercase tracking-wider text-violet-700 dark:text-violet-300">Vector-to-sensor bridge</p>
            <p className="mt-2 text-sm leading-6 text-zinc-700 dark:text-zinc-300">
              VectorWorld supplies an explicit vector-world prior. After projection into image space, a downstream video model can generate sensor-level surround-view videos with stronger geometry, motion, and interaction consistency for autonomous driving.
            </p>
          </div>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-12">
        <StageCard title="Stage 1 · Vector world" badge="World prior" caption="Explicit vector scene generated by VectorWorld." src={current.vectorSrc} tone="zinc" colSpan="xl:col-span-3" />
        <StageCard title="Stage 2 · Sensor-space projection" badge="Bridge" caption="Projection that exposes the structural prior in image space." src={current.layoutSrc} tone="blue" colSpan="xl:col-span-4" />
        <StageCard title="Stage 3 · Sensor-level video" badge="Video model" caption="Surround-view video conditioned on the projected structural prior." src={current.resultSrc} tone="violet" colSpan="xl:col-span-5" />
      </div>

      <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 text-sm leading-6 text-zinc-600 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
        <strong className="text-zinc-900 dark:text-zinc-100">Takeaway.</strong>{" "}
        Vector scene → projected layout → sensor-level video. The vector world model keeps structure explicit, and the video model lifts it to a more physical, controllable surround-view world.
      </div>
    </div>
  );
}