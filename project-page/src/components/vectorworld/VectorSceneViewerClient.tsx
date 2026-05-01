import {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/* ================================================================
   Types
   ================================================================ */

type DType = "f32" | "u8" | "u16";
type Point = { x: number; y: number };

type IndexSceneEntry = {
  id: string;
  path: string;
  num_agents: number;
  num_lanes: number;
  route_points: number;
  route_completed: boolean;
  bbox_xyxy: [number, number, number, number];
};

type SceneIndex = { scenes: IndexSceneEntry[] };

type BufferDescriptor = {
  file: string;
  dtype: DType;
  shape: number[];
  optional?: boolean;
  labels?: string[];
};

type SceneMeta = {
  format: string;
  version: number;
  dataset: string;
  run_name: string;
  scene_id: string;
  units: string;
  endianness: string;
  ego_index: number;
  lane_width_m: number;
  bbox_xyxy: [number, number, number, number];
  counts: {
    num_agents: number;
    num_lanes: number;
    num_points_per_lane: number;
    route_points: number;
    tiles: number;
    lane_edges_sparse: number;
  };
  meta: Record<string, unknown>;
  buffers: {
    agent_states: BufferDescriptor;
    agent_types: BufferDescriptor & { labels?: string[] };
    agent_motion?: BufferDescriptor;
    lanes: BufferDescriptor;
    route: BufferDescriptor;
    tile_corners: BufferDescriptor;
    lane_connections: {
      src: BufferDescriptor;
      dst: BufferDescriptor;
      type: BufferDescriptor;
      enum: Record<string, number>;
    };
  };
};

type LoadedScene = {
  meta: SceneMeta;
  agentStates: Float32Array;
  agentTypes: Uint8Array;
  agentMotion?: Float32Array;
  lanes: Float32Array;
  route: Float32Array;
  tileCorners: Float32Array;
  laneConnSrc: Uint16Array;
  laneConnDst: Uint16Array;
  laneConnType: Uint8Array;
};

type ParsedAgent = {
  id: number;
  x: number;
  y: number;
  heading: number;
  length: number;
  width: number;
  label: string;
  isEgo: boolean;
  motion: Point[];
};

type ParsedConnectionKind = "succ" | "left" | "right" | "other";
type ParsedConnection = {
  srcTail: Point;
  dstHead: Point;
  kind: ParsedConnectionKind;
};

type ParsedScene = {
  meta: SceneMeta;
  agents: ParsedAgent[];
  lanes: Point[][];
  route: Point[];
  tiles: Point[][];
  connections: ParsedConnection[];
};

type HoverState = {
  agent: ParsedAgent;
  screenX: number;
  screenY: number;
};

type ThreePalette = {
  background: number;
  ground: number;
  gridLine: number;
  roadSurface: number;
  laneCenter: number;
  route: number;
  tileFill: number;
  tileOpacity: number;
  connection: number;
  ego: number;
  vehicle: number;
  pedestrian: number;
  cyclist: number;
  trail: number;
  highlight: number;
  headingIndicator: number;
};

type Props = { indexUrl: string };

const STATE_LAYOUT = {
  x: 0,
  y: 1,
  speed: 2,
  cosHeading: 3,
  sinHeading: 4,
  length: 5,
  width: 6,
  legacyHeading: 2,
} as const;

const MOTION_LOOP_SECONDS = 6;
const MOTION_POINT_EPS = 1e-3;
const CURRENT_ANCHOR_THRESHOLD_M = 0.35;
const TRAIL_LOCAL_Y_SIGN: 1 | -1 = 1;
const AGENT_YAW_OFFSET = 0;

const LAYER_Y = {
  ground: -0.02,
  tile: 0.0,
  road: 0.02,
  laneCenter: 0.05,
  connection: 0.04,
  route: 0.07,
  trail: 0.09,
} as const;

const AGENT_H: Record<string, number> = {
  vehicle: 1.5,
  pedestrian: 1.7,
  cyclist: 1.3,
};

/* ================================================================
   Utility helpers
   ================================================================ */

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function toDeg(r: number) {
  return (r * 180) / Math.PI;
}

function wrapAnglePi(angle: number) {
  let a = angle;
  while (a <= -Math.PI) a += Math.PI * 2;
  while (a > Math.PI) a -= Math.PI * 2;
  return a;
}

function resolveHeadingFromAgentState(
  state: Float32Array,
  base: number,
  stride: number,
) {
  const cosHeading =
    STATE_LAYOUT.cosHeading < stride
      ? state[base + STATE_LAYOUT.cosHeading]
      : undefined;

  const sinHeading =
    STATE_LAYOUT.sinHeading < stride
      ? state[base + STATE_LAYOUT.sinHeading]
      : undefined;

  // 推荐路径：VectorWorld 官方统一格式
  if (
    isFiniteNumber(cosHeading) &&
    isFiniteNumber(sinHeading) &&
    Math.abs(cosHeading) <= 1.1 &&
    Math.abs(sinHeading) <= 1.1
  ) {
    const norm = Math.hypot(cosHeading, sinHeading);
    if (norm > 1e-6) {
      return wrapAnglePi(
        Math.atan2(sinHeading / norm, cosHeading / norm),
      );
    }
  }

  // 兼容某些旧格式：如果真的存在“第 3 维就是 heading”的历史导出
  const legacyHeading =
    STATE_LAYOUT.legacyHeading < stride
      ? state[base + STATE_LAYOUT.legacyHeading]
      : undefined;

  if (
    isFiniteNumber(legacyHeading) &&
    Math.abs(legacyHeading) <= Math.PI * 2 + 1e-3
  ) {
    return wrapAnglePi(legacyHeading);
  }

  return 0;
}

function lastDim(shape: number[], fb: number) {
  return shape.length > 0 ? shape[shape.length - 1] : fb;
}

function distPt(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/* ================================================================
   Data loading
   ================================================================ */

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  return (await r.json()) as T;
}

/* ★ FIX 1: 增加第四个重载签名，接受 DType 联合类型 */
async function fetchTypedArray(
  url: string,
  dtype: "f32",
): Promise<Float32Array>;
async function fetchTypedArray(
  url: string,
  dtype: "u8",
): Promise<Uint8Array>;
async function fetchTypedArray(
  url: string,
  dtype: "u16",
): Promise<Uint16Array>;
async function fetchTypedArray(
  url: string,
  dtype: DType,
): Promise<Float32Array | Uint8Array | Uint16Array>;
async function fetchTypedArray(
  url: string,
  dtype: DType,
): Promise<Float32Array | Uint8Array | Uint16Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  const buf = await r.arrayBuffer();
  if (dtype === "f32") return new Float32Array(buf);
  if (dtype === "u8") return new Uint8Array(buf);
  if (dtype === "u16") return new Uint16Array(buf);
  throw new Error(`Unsupported dtype: ${dtype}`);
}

/* ================================================================
   Agent size + motion polyline helpers
   ================================================================ */

function resolveAgentSize(label: string, rawL?: number, rawW?: number) {
  const fb =
    label === "pedestrian"
      ? { length: 0.8, width: 0.8 }
      : label === "cyclist"
        ? { length: 1.8, width: 0.7 }
        : { length: 4.6, width: 1.9 };

  const length =
    isFiniteNumber(rawL) && rawL >= 0.4 && rawL <= 12 ? rawL : fb.length;
  const width =
    isFiniteNumber(rawW) && rawW >= 0.3 && rawW <= 4 ? rawW : fb.width;

  return { length, width };
}

function dedupePolyline(pts: Point[], eps = MOTION_POINT_EPS) {
  const out: Point[] = [];
  pts.forEach((p) => {
    const prev = out[out.length - 1];
    if (!prev || distPt(prev, p) > eps) out.push(p);
  });
  return out;
}

function resamplePolyline(pts: Point[], n: number) {
  if (pts.length < 2 || n <= pts.length) return pts;

  const cum: number[] = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + distPt(pts[i - 1], pts[i]));
  }

  const total = cum[cum.length - 1];
  if (!Number.isFinite(total) || total < 1e-6) return pts;

  const out: Point[] = [];
  for (let s = 0; s < n; s++) {
    const d = (s / Math.max(n - 1, 1)) * total;
    let si = 0;
    while (si < cum.length - 2 && cum[si + 1] < d) si++;
    const segLen = Math.max(cum[si + 1] - cum[si], 1e-6);
    const a = (d - cum[si]) / segLen;
    out.push({
      x: lerp(pts[si].x, pts[si + 1].x, a),
      y: lerp(pts[si].y, pts[si + 1].y, a),
    });
  }

  return dedupePolyline(out, 1e-4);
}

function buildHistoryMotionPolyline(
  x: number,
  y: number,
  heading: number,
  raw: Float32Array | undefined,
  base: number,
  stride: number,
) {
  const cur = { x, y };
  if (!raw || stride < 2) return [cur];

  const pts: Point[] = [];
  for (let k = 0; k < Math.floor(stride / 2); k++) {
    const dx = raw[base + k * 2];
    const dy = TRAIL_LOCAL_Y_SIGN * raw[base + k * 2 + 1];
    if (!isFiniteNumber(dx) || !isFiniteNumber(dy)) continue;
    pts.push({
      x: x + Math.cos(heading) * dx - Math.sin(heading) * dy,
      y: y + Math.sin(heading) * dx + Math.cos(heading) * dy,
    });
  }

  const cleaned = dedupePolyline(pts);
  if (cleaned.length === 0) return [cur];

  const ai = cleaned.reduce(
    (bi, p, i) => (distPt(p, cur) < distPt(cleaned[bi], cur) ? i : bi),
    0,
  );

  const pre = cleaned.slice(0, ai + 1);
  const suf = cleaned.slice(ai).reverse();
  const ordered =
    distPt(cleaned[0], cur) >= distPt(cleaned[cleaned.length - 1], cur)
      ? pre
      : suf;

  const tail = ordered[ordered.length - 1];
  const merged =
    distPt(tail, cur) <= CURRENT_ANCHOR_THRESHOLD_M
      ? [...ordered.slice(0, -1), cur]
      : [...ordered, cur];

  const final = dedupePolyline(merged, 1e-3);
  return resamplePolyline(final, Math.round(clamp(final.length * 4, 12, 48)));
}

function samplePolyline(pts: Point[], t: number) {
  if (pts.length === 0) return null;
  if (pts.length === 1) return pts[0];

  const pos = clamp(t, 0, 1) * (pts.length - 1);
  const i = Math.floor(pos);
  const ni = Math.min(i + 1, pts.length - 1);
  const a = pos - i;

  return {
    x: lerp(pts[i].x, pts[ni].x, a),
    y: lerp(pts[i].y, pts[ni].y, a),
  };
}

function sampleHeading(pts: Point[], t: number) {
  if (pts.length < 2) return null;

  const pos = clamp(t, 0, 0.999999) * (pts.length - 1);
  const i = Math.floor(pos);
  const ni = Math.min(i + 1, pts.length - 1);
  const dx = pts[ni].x - pts[i].x;
  const dy = pts[ni].y - pts[i].y;

  if (Math.abs(dx) + Math.abs(dy) < 1e-6) return null;
  return Math.atan2(dy, dx);
}

function formatAgentLabel(label: string) {
  if (!label) return "Agent";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

/* ================================================================
   Scene parser
   ================================================================ */

function parseScene(scene: LoadedScene): ParsedScene {
  const meta = scene.meta;
  const ptsPerLane =
    meta.buffers.lanes.shape[1] ?? meta.counts.num_points_per_lane;
  const numLanes = Math.min(
    meta.counts.num_lanes,
    Math.floor(scene.lanes.length / Math.max(ptsPerLane * 2, 1)),
  );
  const routeN = Math.min(
    meta.counts.route_points,
    Math.floor(scene.route.length / 2),
  );
  const tileN = Math.min(
    meta.counts.tiles,
    Math.floor(scene.tileCorners.length / 8),
  );

  const stateStride = lastDim(
    meta.buffers.agent_states.shape,
    Math.max(
      1,
      Math.floor(scene.agentStates.length / Math.max(meta.counts.num_agents, 1)),
    ),
  );

  const motionStride =
    scene.agentMotion && meta.buffers.agent_motion
      ? lastDim(
          meta.buffers.agent_motion.shape,
          Math.max(
            0,
            Math.floor(
              scene.agentMotion.length / Math.max(meta.counts.num_agents, 1),
            ),
          ),
        )
      : 0;

  const numAgents = Math.min(
    meta.counts.num_agents,
    scene.agentTypes.length,
    Math.floor(scene.agentStates.length / Math.max(stateStride, 1)),
  );

  const typeLabels =
    meta.buffers.agent_types.labels ?? ["vehicle", "pedestrian", "cyclist"];

  const lanes: Point[][] = [];
  for (let li = 0; li < numLanes; li++) {
    const pts: Point[] = [];
    for (let pi = 0; pi < ptsPerLane; pi++) {
      const b = (li * ptsPerLane + pi) * 2;
      const px = scene.lanes[b];
      const py = scene.lanes[b + 1];
      if (isFiniteNumber(px) && isFiniteNumber(py)) pts.push({ x: px, y: py });
    }
    lanes.push(pts);
  }

  const route: Point[] = [];
  for (let i = 0; i < routeN; i++) {
    const px = scene.route[i * 2];
    const py = scene.route[i * 2 + 1];
    if (isFiniteNumber(px) && isFiniteNumber(py)) route.push({ x: px, y: py });
  }

  const tiles: Point[][] = [];
  for (let t = 0; t < tileN; t++) {
    const corners: Point[] = [];
    for (let c = 0; c < 4; c++) {
      const px = scene.tileCorners[t * 8 + c * 2];
      const py = scene.tileCorners[t * 8 + c * 2 + 1];
      if (isFiniteNumber(px) && isFiniteNumber(py))
        corners.push({ x: px, y: py });
    }
    tiles.push(corners);
  }

  const agents: ParsedAgent[] = [];
  for (let ai = 0; ai < numAgents; ai++) {
    const sb = ai * stateStride;
    const ax = scene.agentStates[sb + STATE_LAYOUT.x];
    const ay = scene.agentStates[sb + STATE_LAYOUT.y];
    if (!isFiniteNumber(ax) || !isFiniteNumber(ay)) continue;

    const hd = resolveHeadingFromAgentState(
      scene.agentStates,
      sb,
      stateStride,
    );
    const ti = scene.agentTypes[ai] ?? 0;
    const label = typeLabels[ti] ?? "vehicle";

    const rL =
      STATE_LAYOUT.length < stateStride
        ? scene.agentStates[sb + STATE_LAYOUT.length]
        : undefined;
    const rW =
      STATE_LAYOUT.width < stateStride
        ? scene.agentStates[sb + STATE_LAYOUT.width]
        : undefined;

    const { length, width } = resolveAgentSize(label, rL, rW);

    const motion =
      scene.agentMotion && motionStride >= 2
        ? buildHistoryMotionPolyline(
            ax,
            ay,
            hd,
            scene.agentMotion,
            ai * motionStride,
            motionStride,
          )
        : [{ x: ax, y: ay }];

    agents.push({
      id: ai,
      x: ax,
      y: ay,
      heading: hd,
      length,
      width,
      label,
      isEgo: ai === meta.ego_index,
      motion,
    });
  }

  const enumMap = meta.buffers.lane_connections.enum;
  const connN = Math.min(
    scene.laneConnSrc.length,
    scene.laneConnDst.length,
    scene.laneConnType.length,
  );

  const connections: ParsedConnection[] = [];
  for (let ci = 0; ci < connN; ci++) {
    const sl = scene.laneConnSrc[ci];
    const dl = scene.laneConnDst[ci];
    const tv = scene.laneConnType[ci];

    if (sl >= lanes.length || dl >= lanes.length) continue;
    if (lanes[sl].length === 0 || lanes[dl].length === 0) continue;

    const kind: ParsedConnectionKind =
      tv === enumMap.succ
        ? "succ"
        : tv === enumMap.left
          ? "left"
          : tv === enumMap.right
            ? "right"
            : "other";

    connections.push({
      srcTail: lanes[sl][lanes[sl].length - 1],
      dstHead: lanes[dl][0],
      kind,
    });
  }

  return { meta, agents, lanes, route, tiles, connections };
}

/* ================================================================
   Theme palette
   ================================================================ */

function isDarkTheme() {
  const t = document.documentElement.dataset.theme;
  if (t === "dark") return true;
  if (t === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function getPalette(dark: boolean): ThreePalette {
  return dark
    ? {
        background: 0x111118,
        ground: 0x18181b,
        gridLine: 0x2a2a30,
        roadSurface: 0x3a3a42,
        laneCenter: 0x808090,
        route: 0x38bdf8,
        tileFill: 0x3b82f6,
        tileOpacity: 0.08,
        connection: 0x22c55e,
        ego: 0xfb923c,
        vehicle: 0xc4c4cc,
        pedestrian: 0x4ade80,
        cyclist: 0xfacc15,
        trail: 0x38bdf8,
        highlight: 0x60a5fa,
        headingIndicator: 0xfafafa,
      }
    : {
        background: 0xf4f4f5,
        ground: 0xe4e4e7,
        gridLine: 0xcccccc,
        roadSurface: 0x6b7280,
        laneCenter: 0xffffff,
        route: 0x0284c7,
        tileFill: 0x3b82f6,
        tileOpacity: 0.1,
        connection: 0x22c55e,
        ego: 0xea580c,
        vehicle: 0x475569,
        pedestrian: 0x16a34a,
        cyclist: 0xca8a04,
        trail: 0x0284c7,
        highlight: 0x2563eb,
        headingIndicator: 0x1e293b,
      };
}

/* ================================================================
   Three.js helpers
   ================================================================ */

function toMaterialArray(
  material: THREE.Material | THREE.Material[],
): THREE.Material[] {
  return Array.isArray(material) ? material : [material];
}

function isDisposableRenderable(
  child: THREE.Object3D,
): child is THREE.Mesh | THREE.Line | THREE.LineSegments {
  return (
    child instanceof THREE.Mesh ||
    child instanceof THREE.Line ||
    child instanceof THREE.LineSegments
  );
}

function disposeGroup(obj: THREE.Object3D) {
  obj.traverse((child: THREE.Object3D) => {
    if (!isDisposableRenderable(child)) return;
    child.geometry.dispose();
    toMaterialArray(child.material).forEach((m: THREE.Material) => {
      m.dispose();
    });
  });
}

function agentBodyColor(agent: ParsedAgent, p: ThreePalette) {
  if (agent.isEgo) return p.ego;
  if (agent.label === "pedestrian") return p.pedestrian;
  if (agent.label === "cyclist") return p.cyclist;
  return p.vehicle;
}

/* ================================================================
   Scene-building functions
   ================================================================ */

function buildGround(
  bbox: [number, number, number, number],
  p: ThreePalette,
): THREE.Group {
  const g = new THREE.Group();
  const [x0, y0, x1, y1] = bbox;
  const cx = (x0 + x1) / 2;
  const cz = -(y0 + y1) / 2;
  const span = Math.max(x1 - x0, y1 - y0, 100) * 2.5;

  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(span, span),
    new THREE.MeshStandardMaterial({
      color: p.ground,
      roughness: 0.95,
      metalness: 0,
      side: THREE.DoubleSide,
    }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(cx, LAYER_Y.ground, cz);
  plane.receiveShadow = true;
  g.add(plane);

  const divs = Math.max(10, Math.floor(span / 10));
  const grid = new THREE.GridHelper(span, divs, p.gridLine, p.gridLine);
  grid.position.set(cx, LAYER_Y.ground + 0.005, cz);

  const gm = toMaterialArray(grid.material);
  gm.forEach((m: THREE.Material) => {
    m.transparent = true;
    m.opacity = 0.25;
  });

  g.add(grid);
  return g;
}

function buildLanes(
  lanes: Point[][],
  laneWidth: number,
  p: ThreePalette,
): THREE.Group {
  const g = new THREE.Group();

  const rPos: number[] = [];
  const rNor: number[] = [];
  const rIdx: number[] = [];
  let vOff = 0;

  for (const lane of lanes) {
    if (lane.length < 2) continue;

    for (let i = 0; i < lane.length; i++) {
      let tx: number;
      let ty: number;

      if (i === 0) {
        tx = lane[1].x - lane[0].x;
        ty = lane[1].y - lane[0].y;
      } else if (i === lane.length - 1) {
        tx = lane[i].x - lane[i - 1].x;
        ty = lane[i].y - lane[i - 1].y;
      } else {
        tx = lane[i + 1].x - lane[i - 1].x;
        ty = lane[i + 1].y - lane[i - 1].y;
      }

      const len = Math.sqrt(tx * tx + ty * ty) || 1;
      const nx = (-ty / len) * laneWidth * 0.5;
      const ny = (tx / len) * laneWidth * 0.5;

      rPos.push(lane[i].x + nx, LAYER_Y.road, -(lane[i].y + ny));
      rPos.push(lane[i].x - nx, LAYER_Y.road, -(lane[i].y - ny));
      rNor.push(0, 1, 0, 0, 1, 0);
    }

    for (let i = 0; i < lane.length - 1; i++) {
      const b = vOff + i * 2;
      rIdx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
    }

    vOff += lane.length * 2;
  }

  if (rPos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(rPos, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(rNor, 3));
    geo.setIndex(rIdx);
    g.add(
      new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({
          color: p.roadSurface,
          roughness: 0.85,
          side: THREE.DoubleSide,
        }),
      ),
    );
  }

  const lPos: number[] = [];
  for (const lane of lanes) {
    for (let i = 0; i < lane.length - 1; i++) {
      lPos.push(lane[i].x, LAYER_Y.laneCenter, -lane[i].y);
      lPos.push(lane[i + 1].x, LAYER_Y.laneCenter, -lane[i + 1].y);
    }
  }

  if (lPos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(lPos, 3));
    g.add(
      new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: p.laneCenter,
          transparent: true,
          opacity: 0.55,
        }),
      ),
    );
  }

  return g;
}

function buildRoute(route: Point[], p: ThreePalette): THREE.Group {
  const g = new THREE.Group();
  if (route.length < 2) return g;

  const pos: number[] = [];
  for (const pt of route) pos.push(pt.x, LAYER_Y.route, -pt.y);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: p.route })));

  const rP: number[] = [];
  const rN: number[] = [];
  const rI: number[] = [];
  const w = 1.4;

  for (let i = 0; i < route.length; i++) {
    let tx: number;
    let ty: number;

    if (i === 0) {
      tx = route[1].x - route[0].x;
      ty = route[1].y - route[0].y;
    } else if (i === route.length - 1) {
      tx = route[i].x - route[i - 1].x;
      ty = route[i].y - route[i - 1].y;
    } else {
      tx = route[i + 1].x - route[i - 1].x;
      ty = route[i + 1].y - route[i - 1].y;
    }

    const len = Math.sqrt(tx * tx + ty * ty) || 1;
    const nx = (-ty / len) * w;
    const ny = (tx / len) * w;

    rP.push(route[i].x + nx, LAYER_Y.route - 0.005, -(route[i].y + ny));
    rP.push(route[i].x - nx, LAYER_Y.route - 0.005, -(route[i].y - ny));
    rN.push(0, 1, 0, 0, 1, 0);
  }

  for (let i = 0; i < route.length - 1; i++) {
    const b = i * 2;
    rI.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
  }

  if (rP.length) {
    const geo2 = new THREE.BufferGeometry();
    geo2.setAttribute("position", new THREE.Float32BufferAttribute(rP, 3));
    geo2.setAttribute("normal", new THREE.Float32BufferAttribute(rN, 3));
    geo2.setIndex(rI);
    g.add(
      new THREE.Mesh(
        geo2,
        new THREE.MeshBasicMaterial({
          color: p.route,
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      ),
    );
  }

  return g;
}

function buildTiles(tiles: Point[][], p: ThreePalette): THREE.Group {
  const g = new THREE.Group();
  const vPos: number[] = [];
  const vNor: number[] = [];
  const vIdx: number[] = [];
  let off = 0;

  for (const tile of tiles) {
    if (tile.length < 3) continue;
    for (let c = 0; c < tile.length; c++) {
      vPos.push(tile[c].x, LAYER_Y.tile, -tile[c].y);
      vNor.push(0, 1, 0);
    }
    if (tile.length >= 3) {
      vIdx.push(off, off + 1, off + 2);
      if (tile.length >= 4) vIdx.push(off, off + 2, off + 3);
    }
    off += tile.length;
  }

  if (vPos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vPos, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(vNor, 3));
    geo.setIndex(vIdx);
    g.add(
      new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({
          color: p.tileFill,
          opacity: p.tileOpacity,
          transparent: true,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      ),
    );
  }

  const ePos: number[] = [];
  for (const tile of tiles) {
    if (tile.length < 3) continue;
    for (let c = 0; c < tile.length; c++) {
      const nc = (c + 1) % tile.length;
      ePos.push(tile[c].x, LAYER_Y.tile + 0.01, -tile[c].y);
      ePos.push(tile[nc].x, LAYER_Y.tile + 0.01, -tile[nc].y);
    }
  }

  if (ePos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(ePos, 3));
    g.add(
      new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: p.tileFill,
          transparent: true,
          opacity: 0.35,
        }),
      ),
    );
  }

  return g;
}

function buildConnections(
  conns: ParsedConnection[],
  p: ThreePalette,
): THREE.Group {
  const g = new THREE.Group();
  const pos: number[] = [];

  for (const c of conns) {
    if (c.kind === "other") continue;
    pos.push(c.srcTail.x, LAYER_Y.connection, -c.srcTail.y);
    pos.push(c.dstHead.x, LAYER_Y.connection, -c.dstHead.y);
  }

  if (pos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.add(
      new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: p.connection,
          transparent: true,
          opacity: 0.5,
        }),
      ),
    );
  }

  return g;
}

type AgentBuildResult = {
  group: THREE.Group;
  meshes: THREE.Mesh[];
};

function buildAgents(agents: ParsedAgent[], p: ThreePalette): AgentBuildResult {
  const group = new THREE.Group();
  const meshes: THREE.Mesh[] = [];

  for (const ag of agents) {
    const h = AGENT_H[ag.label] ?? 1.5;
    const col = agentBodyColor(ag, p);

    const box = new THREE.Mesh(
      new THREE.BoxGeometry(ag.length, h, ag.width),
      new THREE.MeshStandardMaterial({
        color: col,
        roughness: 0.5,
        metalness: 0.15,
      }),
    );
    box.position.set(ag.x, h / 2, -ag.y);
    box.rotation.y = ag.heading + AGENT_YAW_OFFSET;
    box.castShadow = true;
    box.userData = { agentData: ag };
    group.add(box);
    meshes.push(box);

    const sr = Math.max(ag.length, ag.width) * 0.55;
    const shadow = new THREE.Mesh(
      new THREE.CircleGeometry(sr, 16),
      new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
      }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(ag.x, 0.003, -ag.y);
    group.add(shadow);

    const coneH = ag.length * 0.18;
    const coneR = ag.width * 0.18;
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(coneR, coneH, 6),
      new THREE.MeshStandardMaterial({ color: p.headingIndicator }),
    );
    cone.geometry.rotateZ(-Math.PI / 2);
    cone.position.set(ag.length * 0.38, h * 0.72, 0);
    box.add(cone);
  }

  return { group, meshes };
}

type TrailBuildResult = {
  linesGroup: THREE.Group;
  ghostGroup: THREE.Group;
  ghostEntries: { mesh: THREE.Mesh; agent: ParsedAgent }[];
};

function buildTrails(agents: ParsedAgent[], p: ThreePalette): TrailBuildResult {
  const linesGroup = new THREE.Group();
  const ghostGroup = new THREE.Group();
  const ghostEntries: { mesh: THREE.Mesh; agent: ParsedAgent }[] = [];

  const pos: number[] = [];
  for (const ag of agents) {
    if (ag.motion.length < 2) continue;
    for (let i = 0; i < ag.motion.length - 1; i++) {
      pos.push(ag.motion[i].x, LAYER_Y.trail, -ag.motion[i].y);
      pos.push(ag.motion[i + 1].x, LAYER_Y.trail, -ag.motion[i + 1].y);
    }
  }

  if (pos.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    linesGroup.add(
      new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: p.trail,
          transparent: true,
          opacity: 0.45,
        }),
      ),
    );
  }

  for (const ag of agents) {
    if (ag.motion.length < 2) continue;
    const h = AGENT_H[ag.label] ?? 1.5;
    const col = agentBodyColor(ag, p);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(ag.length, h, ag.width),
      new THREE.MeshStandardMaterial({
        color: col,
        transparent: true,
        opacity: 0.32,
        depthWrite: false,
      }),
    );
    mesh.visible = false;
    ghostGroup.add(mesh);
    ghostEntries.push({ mesh, agent: ag });
  }

  return { linesGroup, ghostGroup, ghostEntries };
}

function buildHighlight(p: ThreePalette): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({
      color: p.highlight,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  mesh.visible = false;
  mesh.renderOrder = 999;
  return mesh;
}

function fitCamera(
  cam: THREE.PerspectiveCamera,
  ctrl: OrbitControls,
  bbox: [number, number, number, number],
  mode: "perspective" | "top" = "perspective",
) {
  const [x0, y0, x1, y1] = bbox;
  const cx = (x0 + x1) / 2;
  const cz = -(y0 + y1) / 2;
  const ext = Math.max(x1 - x0, y1 - y0, 50);
  const target = new THREE.Vector3(cx, 0, cz);

  ctrl.target.copy(target);
  if (mode === "top") {
    cam.position.set(cx, ext * 1.0, cz + 0.01);
  } else {
    cam.position.set(cx + ext * 0.12, ext * 0.55, cz + ext * 0.35);
  }
  cam.lookAt(target);
  ctrl.update();
}

/* ================================================================
   React component
   ================================================================ */

type ThreeState = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  rafId: number;
  root: THREE.Group | null;
  routeGroup: THREE.Group | null;
  tilesGroup: THREE.Group | null;
  connsGroup: THREE.Group | null;
  trailLinesGroup: THREE.Group | null;
  ghostGroup: THREE.Group | null;
  agentMeshes: THREE.Mesh[];
  ghostEntries: { mesh: THREE.Mesh; agent: ParsedAgent }[];
  highlight: THREE.Mesh | null;
  isDragging: boolean;
};

export default function VectorSceneViewerClient({ indexUrl }: Props) {
  const [indexData, setIndexData] = useState<SceneIndex | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [sceneData, setSceneData] = useState<LoadedScene | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [themeVer, setThemeVer] = useState(0);

  const [showRoute, setShowRoute] = useState(true);
  const [showTiles, setShowTiles] = useState(true);
  const [showConnections, setShowConnections] = useState(false);
  const [showMotion, setShowMotion] = useState(true);
  const [animateMotion, setAnimateMotion] = useState(false);
  const [motionProgress, setMotionProgress] = useState(0.35);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);

  const mountRef = useRef<HTMLDivElement>(null);
  const threeRef = useRef<ThreeState | null>(null);

  const parsedScene = useMemo(
    () => (sceneData ? parseScene(sceneData) : null),
    [sceneData],
  );

  const selectedEntry = useMemo(
    () => indexData?.scenes.find((s) => s.id === selectedId) ?? null,
    [indexData, selectedId],
  );

  const sceneIdx = useMemo(
    () => indexData?.scenes.findIndex((s) => s.id === selectedId) ?? -1,
    [indexData, selectedId],
  );

  const totalScenes = indexData?.scenes.length ?? 0;

  const goPrevScene = useCallback(() => {
    if (!indexData || totalScenes <= 1) return;
    const ni = sceneIdx <= 0 ? totalScenes - 1 : sceneIdx - 1;
    setSelectedId(indexData.scenes[ni].id);
  }, [indexData, sceneIdx, totalScenes]);

  const goNextScene = useCallback(() => {
    if (!indexData || totalScenes <= 1) return;
    const ni = sceneIdx >= totalScenes - 1 ? 0 : sceneIdx + 1;
    setSelectedId(indexData.scenes[ni].id);
  }, [indexData, sceneIdx, totalScenes]);

  useEffect(() => {
    let cancel = false;

    (async () => {
      try {
        const url = new URL(indexUrl, window.location.href).toString();
        const data = await fetchJson<SceneIndex>(url);
        if (cancel) return;
        setIndexData(data);
        if (data.scenes.length > 0) {
          setSelectedId((c) => c || data.scenes[0].id);
        }
      } catch (e) {
        if (!cancel) {
          setError(e instanceof Error ? e.message : "Failed to load index.");
        }
      }
    })();

    return () => {
      cancel = true;
    };
  }, [indexUrl]);

  useEffect(() => {
    if (!indexData || !selectedId) return;

    const entry = indexData.scenes.find((s) => s.id === selectedId);
    if (!entry) return;

    const absIndex = new URL(indexUrl, window.location.href).toString();
    let cancel = false;

    (async () => {
      setLoading(true);
      setError("");
      setSceneData(null);
      setHoverState(null);

      try {
        const sceneUrl = new URL(entry.path, absIndex).toString();
        const meta = await fetchJson<SceneMeta>(sceneUrl);
        const b = meta.buffers;
        const u = (f: string, d: DType) =>
          fetchTypedArray(new URL(f, sceneUrl).toString(), d);

        /* ★ FIX 2: as unknown as [...] 双重断言 */
        const [
          agentStates,
          agentTypes,
          lanes,
          route,
          tileCorners,
          lcs,
          lcd,
          lct,
          agentMotion,
        ] = (await Promise.all([
          u(b.agent_states.file, b.agent_states.dtype),
          u(b.agent_types.file, b.agent_types.dtype),
          u(b.lanes.file, b.lanes.dtype),
          u(b.route.file, b.route.dtype),
          u(b.tile_corners.file, b.tile_corners.dtype),
          u(b.lane_connections.src.file, b.lane_connections.src.dtype),
          u(b.lane_connections.dst.file, b.lane_connections.dst.dtype),
          u(b.lane_connections.type.file, b.lane_connections.type.dtype),
          b.agent_motion
            ? u(b.agent_motion.file, b.agent_motion.dtype)
            : Promise.resolve(undefined),
        ])) as unknown as [
          Float32Array,
          Uint8Array,
          Float32Array,
          Float32Array,
          Float32Array,
          Uint16Array,
          Uint16Array,
          Uint8Array,
          Float32Array | undefined,
        ];

        if (cancel) return;

        setSceneData({
          meta,
          agentStates,
          agentTypes,
          agentMotion,
          lanes,
          route,
          tileCorners,
          laneConnSrc: lcs,
          laneConnDst: lcd,
          laneConnType: lct,
        });
      } catch (e) {
        if (!cancel) {
          setError(e instanceof Error ? e.message : "Failed to load scene.");
        }
      } finally {
        if (!cancel) setLoading(false);
      }
    })();

    return () => {
      cancel = true;
    };
  }, [indexData, selectedId, indexUrl]);

  useEffect(() => {
    const bump = () => setThemeVer((v) => v + 1);

    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", bump);

    return () => {
      mo.disconnect();
      mq.removeEventListener("change", bump);
    };
  }, []);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = false;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(
      50,
      mount.clientWidth / Math.max(mount.clientHeight, 1),
      0.1,
      5000,
    );
    cam.position.set(0, 200, 100);

    const ctrl = new OrbitControls(cam, renderer.domElement);
    ctrl.enableDamping = true;
    ctrl.dampingFactor = 0.08;
    ctrl.screenSpacePanning = true;
    ctrl.minDistance = 3;
    ctrl.maxDistance = 3000;
    ctrl.maxPolarAngle = Math.PI / 2 - 0.01;

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dLight = new THREE.DirectionalLight(0xffffff, 0.7);
    dLight.position.set(60, 180, 40);
    scene.add(dLight);

    const st: ThreeState = {
      renderer,
      scene,
      camera: cam,
      controls: ctrl,
      rafId: 0,
      root: null,
      routeGroup: null,
      tilesGroup: null,
      connsGroup: null,
      trailLinesGroup: null,
      ghostGroup: null,
      agentMeshes: [],
      ghostEntries: [],
      highlight: null,
      isDragging: false,
    };
    threeRef.current = st;

    ctrl.addEventListener("start", () => {
      st.isDragging = true;
    });
    ctrl.addEventListener("end", () => {
      st.isDragging = false;
    });

    function animate() {
      st.rafId = requestAnimationFrame(animate);
      ctrl.update();
      renderer.render(scene, cam);
    }
    animate();

    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(st.rafId);
      ro.disconnect();

      if (st.root) {
        st.scene.remove(st.root);
        disposeGroup(st.root);
        st.root = null;
      }

      ctrl.dispose();
      renderer.dispose();

      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }

      threeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const st = threeRef.current;
    if (!st || !parsedScene) return;

    if (st.root) {
      st.scene.remove(st.root);
      disposeGroup(st.root);
      st.root = null;
    }

    const dark = isDarkTheme();
    const p = getPalette(dark);
    st.scene.background = new THREE.Color(p.background);

    const root = new THREE.Group();
    const lw = isFiniteNumber(parsedScene.meta.lane_width_m)
      ? parsedScene.meta.lane_width_m
      : 4.2;

    const ground = buildGround(parsedScene.meta.bbox_xyxy, p);
    const lanes = buildLanes(parsedScene.lanes, lw, p);
    const routeG = buildRoute(parsedScene.route, p);
    const tilesG = buildTiles(parsedScene.tiles, p);
    const connsG = buildConnections(parsedScene.connections, p);
    const { group: agGroup, meshes: agMeshes } = buildAgents(
      parsedScene.agents,
      p,
    );
    const { linesGroup, ghostGroup, ghostEntries } = buildTrails(
      parsedScene.agents,
      p,
    );
    const hl = buildHighlight(p);

    root.add(
      ground,
      lanes,
      routeG,
      tilesG,
      connsG,
      agGroup,
      linesGroup,
      ghostGroup,
      hl,
    );
    st.scene.add(root);

    st.root = root;
    st.routeGroup = routeG;
    st.tilesGroup = tilesG;
    st.connsGroup = connsG;
    st.trailLinesGroup = linesGroup;
    st.ghostGroup = ghostGroup;
    st.agentMeshes = agMeshes;
    st.ghostEntries = ghostEntries;
    st.highlight = hl;

    fitCamera(
      st.camera,
      st.controls,
      parsedScene.meta.bbox_xyxy,
      "perspective",
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parsedScene, themeVer]);

  useEffect(() => {
    const st = threeRef.current;
    if (!st) return;
    if (st.routeGroup) st.routeGroup.visible = showRoute;
    if (st.tilesGroup) st.tilesGroup.visible = showTiles;
    if (st.connsGroup) st.connsGroup.visible = showConnections;
    if (st.trailLinesGroup) st.trailLinesGroup.visible = showMotion;
    if (st.ghostGroup) st.ghostGroup.visible = showMotion;
  }, [showRoute, showTiles, showConnections, showMotion]);

  useEffect(() => {
    const st = threeRef.current;
    if (!st) return;

    for (const { mesh, agent } of st.ghostEntries) {
      if (agent.motion.length < 2) {
        mesh.visible = false;
        continue;
      }

      const pos = samplePolyline(agent.motion, motionProgress);
      const hd = sampleHeading(agent.motion, motionProgress) ?? agent.heading;

      if (pos) {
        const h = AGENT_H[agent.label] ?? 1.5;
        mesh.position.set(pos.x, h / 2, -pos.y);
        mesh.rotation.y = hd + AGENT_YAW_OFFSET;
        mesh.visible = true;
      } else {
        mesh.visible = false;
      }
    }
  }, [motionProgress]);

  useEffect(() => {
    const mount = mountRef.current;
    const st = threeRef.current;
    if (!mount || !st) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMove = (e: PointerEvent) => {
      if (st.isDragging) {
        setHoverState(null);
        if (st.highlight) st.highlight.visible = false;
        return;
      }

      const rect = mount.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      mouse.x = (sx / rect.width) * 2 - 1;
      mouse.y = -(sy / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, st.camera);
      const hits = raycaster.intersectObjects(st.agentMeshes);

      if (hits.length > 0) {
        const ag = hits[0].object.userData.agentData as ParsedAgent;

        if (st.highlight) {
          const h = AGENT_H[ag.label] ?? 1.5;
          st.highlight.scale.set(
            ag.length * 1.15,
            h * 1.15,
            ag.width * 1.15,
          );
          st.highlight.position.set(ag.x, h / 2, -ag.y);
          st.highlight.rotation.y = ag.heading + AGENT_YAW_OFFSET;
          st.highlight.visible = true;
        }

        setHoverState({ agent: ag, screenX: sx, screenY: sy });
      } else {
        if (st.highlight) st.highlight.visible = false;
        setHoverState(null);
      }
    };

    const onLeave = () => {
      setHoverState(null);
      if (st.highlight) st.highlight.visible = false;
    };

    mount.addEventListener("pointermove", onMove);
    mount.addEventListener("pointerleave", onLeave);

    return () => {
      mount.removeEventListener("pointermove", onMove);
      mount.removeEventListener("pointerleave", onLeave);
    };
  }, [parsedScene]);

  useEffect(() => {
    if (!animateMotion) return;

    let raf = 0;
    let last: number | null = null;

    const tick = (ts: number) => {
      if (last === null) last = ts;
      const dt = (ts - last) / 1000;
      last = ts;
      setMotionProgress((c) => {
        const n = c + dt / MOTION_LOOP_SECONDS;
        return n >= 1 ? n - Math.floor(n) : n;
      });
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animateMotion]);

  const doFit = useCallback(
    (mode: "perspective" | "top" = "perspective") => {
      const st = threeRef.current;
      if (!st || !parsedScene) return;
      fitCamera(st.camera, st.controls, parsedScene.meta.bbox_xyxy, mode);
      setHoverState(null);
    },
    [parsedScene],
  );

  const centerEgo = useCallback(() => {
    const st = threeRef.current;
    if (!st || !parsedScene) return;

    const ego =
      parsedScene.agents.find((a) => a.isEgo) ?? parsedScene.agents[0];
    if (!ego) return;

    const target = new THREE.Vector3(ego.x, 0, -ego.y);
    st.controls.target.copy(target);
    st.controls.update();
  }, [parsedScene]);

  const chip =
    "rounded-full bg-zinc-100 px-3 py-1 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300";
  const btnSec =
    "rounded-full border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800";
  const btnPri =
    "rounded-full bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white";
  const btnBlue =
    "rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-400";

  const containerH = "clamp(30rem, 74vh, 46rem)";

  const mountW = mountRef.current?.clientWidth ?? 600;
  const mountH = mountRef.current?.clientHeight ?? 400;
  const ttLeft = hoverState
    ? Math.max(12, Math.min(hoverState.screenX + 16, mountW - 240))
    : 0;
  const ttTop = hoverState
    ? Math.max(12, Math.min(hoverState.screenY - 12, mountH - 160))
    : 0;

  return (
    <div className="space-y-5">
      <section className="rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="grid gap-6 xl:grid-cols-2">
          <div className="space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Scene
              </span>
              <select
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
                className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
              >
                {indexData?.scenes.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id}
                  </option>
                ))}
              </select>
            </label>

            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={goPrevScene} className={btnSec}>
                ← Previous
              </button>
              <button type="button" onClick={goNextScene} className={btnPri}>
                Next →
              </button>
              <span className={chip}>
                {sceneIdx + 1} / {totalScenes}
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => doFit("perspective")}
                className={btnPri}
              >
                Fit view
              </button>
              <button
                type="button"
                onClick={() => doFit("top")}
                className={btnSec}
              >
                Top view
              </button>
              <button type="button" onClick={centerEgo} className={btnSec}>
                Center ego
              </button>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                3D Inspection
              </h3>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Orbit, pan, and zoom the exported vector scene in 3D
              </p>
              <ul className="mt-3 space-y-1.5 text-sm text-zinc-600 dark:text-zinc-400">
                <li>🖱️ Left drag → orbit</li>
                <li>🖱️ Right drag → pan</li>
                <li>🖱️ Scroll → zoom (no page scroll)</li>
                <li>🖱️ Double-click → fit view</li>
              </ul>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Layers
              </h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {(
                  [
                    ["Route", showRoute, setShowRoute],
                    ["Tile bounds", showTiles, setShowTiles],
                    ["Lane links", showConnections, setShowConnections],
                    [
                      "Motion trail",
                      showMotion,
                      (v: boolean) => {
                        setShowMotion(v);
                        if (!v) setAnimateMotion(false);
                      },
                    ],
                  ] as const
                ).map(([label, val, setter]) => (
                  <label
                    key={label}
                    className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300"
                  >
                    <input
                      type="checkbox"
                      checked={val as boolean}
                      onChange={(e) =>
                        (setter as (v: boolean) => void)(e.target.checked)
                      }
                      className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="mb-2 flex items-center justify-between text-sm font-medium text-zinc-700 dark:text-zinc-300">
                <span>Motion trail progress</span>
                <span>{Math.round(motionProgress * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1000}
                step={1}
                value={Math.round(motionProgress * 1000)}
                disabled={!showMotion}
                onChange={(e) => {
                  setAnimateMotion(false);
                  setMotionProgress(Number(e.target.value) / 1000);
                }}
                className="w-full accent-blue-600"
              />
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={!showMotion}
                  onClick={() => setAnimateMotion((c) => !c)}
                  className={btnBlue}
                >
                  {animateMotion ? "Pause trail" : "Play trail"}
                </button>
                <button
                  type="button"
                  disabled={!showMotion}
                  onClick={() => {
                    setAnimateMotion(false);
                    setMotionProgress(0);
                  }}
                  className={btnSec}
                >
                  Reset trail
                </button>
              </div>
            </div>
          </div>
        </div>

        {selectedEntry && (
          <div className="mt-5 flex flex-wrap gap-2">
            <span className={chip}>scene {selectedEntry.id}</span>
            <span className={chip}>{sceneData?.meta.dataset ?? "dataset"}</span>
            <span className={chip}>{selectedEntry.num_lanes} lanes</span>
            <span className={chip}>{selectedEntry.num_agents} agents</span>
            <span className={chip}>{selectedEntry.route_points} route pts</span>
            <span
              className={
                selectedEntry.route_completed
                  ? "rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 dark:bg-green-500/10 dark:text-green-300"
                  : chip
              }
            >
              {selectedEntry.route_completed
                ? "route completed"
                : "route not completed"}
            </span>
          </div>
        )}
      </section>

      <div
        ref={mountRef}
        className="relative w-full select-none overflow-hidden rounded-3xl border border-zinc-200 bg-zinc-950 shadow-sm dark:border-zinc-800"
        style={{ height: containerH }}
        onDoubleClick={(e) => {
          e.preventDefault();
          doFit("perspective");
        }}
      >
        <div className="pointer-events-none absolute left-4 top-4 z-10 flex flex-wrap gap-2 text-xs text-zinc-300">
          <span className="rounded-full border border-zinc-700 bg-zinc-900/90 px-3 py-1 backdrop-blur">
            3D interactive canvas
          </span>
          <span className="rounded-full border border-zinc-700 bg-zinc-900/90 px-3 py-1 backdrop-blur">
            Double-click to fit
          </span>
        </div>

        {(loading || !sceneData) && !error && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-zinc-950/80 text-sm font-medium text-zinc-300 backdrop-blur">
            Loading vector scene…
          </div>
        )}
        {error && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-zinc-950/85 px-6 text-center text-sm font-medium text-red-400 backdrop-blur">
            {error}
          </div>
        )}

        {hoverState && (
          <div
            className="pointer-events-none absolute z-30 w-56 rounded-2xl border border-zinc-700 bg-zinc-900/95 p-3 text-xs shadow-lg backdrop-blur"
            style={{ left: ttLeft, top: ttTop }}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="font-semibold text-zinc-100">
                agent {hoverState.agent.id}
              </div>
              <div className="rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-300">
                {hoverState.agent.isEgo
                  ? "Ego"
                  : formatAgentLabel(hoverState.agent.label)}
              </div>
            </div>
            <div className="mt-3 space-y-1.5 text-zinc-400">
              <div>
                position ({hoverState.agent.x.toFixed(1)},{" "}
                {hoverState.agent.y.toFixed(1)}) m
              </div>
              <div>heading {toDeg(hoverState.agent.heading).toFixed(1)}°</div>
              <div>
                bbox {hoverState.agent.length.toFixed(1)} ×{" "}
                {hoverState.agent.width.toFixed(1)} m
              </div>
              <div>
                {Math.max(0, hoverState.agent.motion.length - 1)} trail samples
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-4 text-xs text-zinc-600 dark:text-zinc-400">
        {(
          [
            ["bg-zinc-400", "Lane band"],
            ["bg-sky-500", "Route"],
            ["bg-sky-400", "Motion trail"],
            ["bg-orange-500", "Ego"],
            ["bg-slate-500", "Vehicle"],
            ["bg-green-500", "Pedestrian"],
            ["bg-amber-500", "Cyclist"],
          ] as const
        ).map(([c, l]) => (
          <span key={l} className="inline-flex items-center gap-2">
            <span className={`h-3 w-3 rounded-full ${c}`} />
            {l}
          </span>
        ))}
      </div>

      <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 text-sm leading-6 text-zinc-600 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
        <strong className="text-zinc-900 dark:text-zinc-100">Note.</strong>{" "}
        Exported scene inspection only: one scene snapshot plus a qualitative{" "}
        <code>agent_motion</code> trail, not a full per-step replay. Drag to
        orbit, scroll to zoom, right-drag to pan.
      </div>
    </div>
  );
}