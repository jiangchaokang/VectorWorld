import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";

type DType = "f32" | "u8" | "u16";

type Point = {
  x: number;
  y: number;
};

type Camera = {
  scale: number;
  fitScale: number;
  tx: number;
  ty: number;
};

type IndexSceneEntry = {
  id: string;
  path: string;
  num_agents: number;
  num_lanes: number;
  route_points: number;
  route_completed: boolean;
  bbox_xyxy: [number, number, number, number];
};

type SceneIndex = {
  scenes: IndexSceneEntry[];
};

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

type Palette = {
  background: string;
  roadFill: string;
  laneCenter: string;
  routeGlow: string;
  route: string;
  tileFill: string;
  tileStroke: string;
  successor: string;
  lateral: string;
  vehicle: string;
  pedestrian: string;
  cyclist: string;
  ego: string;
  roofLight: string;
  outline: string;
  heading: string;
  hover: string;
  motionVehicle: string;
  motionPedestrian: string;
  motionCyclist: string;
  agentShadow: string;
};

type Props = {
  indexUrl: string;
};

const STATE_LAYOUT = {
  x: 0,
  y: 1,
  heading: 2,
  length: 5,
  width: 6,
} as const;

const MOTION_LOOP_SECONDS = 6;
const MOTION_POINT_EPS = 1e-3;
const CURRENT_ANCHOR_THRESHOLD_M = 0.35;

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}`);
  }
  return (await response.json()) as T;
}

async function fetchTypedArray(url: string, dtype: DType) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}`);
  }

  const buffer = await response.arrayBuffer();

  switch (dtype) {
    case "f32":
      return new Float32Array(buffer);
    case "u8":
      return new Uint8Array(buffer);
    case "u16":
      return new Uint16Array(buffer);
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function isDarkTheme() {
  const theme = document.documentElement.dataset.theme;
  if (theme === "dark") return true;
  if (theme === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clamp(value: number, minValue: number, maxValue: number) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function toDegrees(radian: number) {
  return (radian * 180) / Math.PI;
}

function lastShapeDim(shape: number[], fallback: number) {
  return shape.length > 0 ? shape[shape.length - 1] : fallback;
}

function createFitCamera(
  bbox: [number, number, number, number],
  width: number,
  height: number,
  padding = 48,
): Camera {
  const [x0, y0, x1, y1] = bbox;
  const bboxWidth = Math.max(x1 - x0, 1);
  const bboxHeight = Math.max(y1 - y0, 1);

  const scale = Math.min(
    Math.max((width - padding * 2) / bboxWidth, 0.001),
    Math.max((height - padding * 2) / bboxHeight, 0.001),
  );

  const centerX = (x0 + x1) / 2;
  const centerY = (y0 + y1) / 2;

  return {
    scale,
    fitScale: scale,
    tx: width / 2 - centerX * scale,
    ty: height / 2 + centerY * scale,
  };
}

function worldToScreen(x: number, y: number, camera: Camera): Point {
  return {
    x: x * camera.scale + camera.tx,
    y: -y * camera.scale + camera.ty,
  };
}

function screenToWorld(x: number, y: number, camera: Camera): Point {
  return {
    x: (x - camera.tx) / camera.scale,
    y: (camera.ty - y) / camera.scale,
  };
}

function resolveAgentSize(label: string, rawLength?: number, rawWidth?: number) {
  const fallback =
    label === "pedestrian"
      ? { length: 0.8, width: 0.8 }
      : label === "cyclist"
        ? { length: 1.8, width: 0.7 }
        : { length: 4.6, width: 1.9 };

  const length =
    isFiniteNumber(rawLength) && rawLength >= 0.4 && rawLength <= 12
      ? rawLength
      : fallback.length;

  const width =
    isFiniteNumber(rawWidth) && rawWidth >= 0.3 && rawWidth <= 4
      ? rawWidth
      : fallback.width;

  return { length, width };
}

function distanceBetweenPoints(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dedupePolyline(points: Point[], epsilon = MOTION_POINT_EPS) {
  const result: Point[] = [];

  points.forEach((point) => {
    const previous = result[result.length - 1];
    if (!previous || distanceBetweenPoints(previous, point) > epsilon) {
      result.push(point);
    }
  });

  return result;
}

function resamplePolyline(points: Point[], targetSamples: number) {
  if (points.length < 2 || targetSamples <= points.length) {
    return points;
  }

  const cumulative: number[] = [0];

  for (let index = 1; index < points.length; index += 1) {
    cumulative.push(
      cumulative[index - 1] + distanceBetweenPoints(points[index - 1], points[index]),
    );
  }

  const totalLength = cumulative[cumulative.length - 1];
  if (!Number.isFinite(totalLength) || totalLength < 1e-6) {
    return points;
  }

  const sampled: Point[] = [];

  for (let sampleIndex = 0; sampleIndex < targetSamples; sampleIndex += 1) {
    const targetDistance =
      (sampleIndex / Math.max(targetSamples - 1, 1)) * totalLength;

    let segmentIndex = 0;
    while (
      segmentIndex < cumulative.length - 2 &&
      cumulative[segmentIndex + 1] < targetDistance
    ) {
      segmentIndex += 1;
    }

    const segmentStartDistance = cumulative[segmentIndex];
    const segmentEndDistance = cumulative[segmentIndex + 1];
    const segmentLength = Math.max(segmentEndDistance - segmentStartDistance, 1e-6);
    const alpha = (targetDistance - segmentStartDistance) / segmentLength;

    sampled.push({
      x: lerp(points[segmentIndex].x, points[segmentIndex + 1].x, alpha),
      y: lerp(points[segmentIndex].y, points[segmentIndex + 1].y, alpha),
    });
  }

  return dedupePolyline(sampled, 1e-4);
}

function buildHistoryMotionPolyline(
  x: number,
  y: number,
  heading: number,
  rawMotion: Float32Array | undefined,
  motionBase: number,
  motionStride: number,
) {
  const currentPoint = { x, y };

  if (!rawMotion || motionStride < 2) {
    return [currentPoint];
  }

  const transformed: Point[] = [];

  for (let k = 0; k < Math.floor(motionStride / 2); k += 1) {
    const dx = rawMotion[motionBase + k * 2];
    const dy = rawMotion[motionBase + k * 2 + 1];

    if (!isFiniteNumber(dx) || !isFiniteNumber(dy)) continue;

    transformed.push({
      x: x + Math.cos(heading) * dx - Math.sin(heading) * dy,
      y: y + Math.sin(heading) * dx + Math.cos(heading) * dy,
    });
  }

  const cleaned = dedupePolyline(transformed);
  if (cleaned.length === 0) {
    return [currentPoint];
  }

  const anchorIndex = cleaned.reduce((bestIndex, point, index) => {
    const bestDistance = distanceBetweenPoints(cleaned[bestIndex], currentPoint);
    const currentDistance = distanceBetweenPoints(point, currentPoint);
    return currentDistance < bestDistance ? index : bestIndex;
  }, 0);

  const prefixEndingAtAnchor = cleaned.slice(0, anchorIndex + 1);
  const suffixEndingAtAnchor = cleaned.slice(anchorIndex).reverse();

  const firstDistance = distanceBetweenPoints(cleaned[0], currentPoint);
  const lastDistance = distanceBetweenPoints(cleaned[cleaned.length - 1], currentPoint);

  const orderedHistory =
    firstDistance >= lastDistance ? prefixEndingAtAnchor : suffixEndingAtAnchor;

  const historyTail = orderedHistory[orderedHistory.length - 1];
  const merged =
    distanceBetweenPoints(historyTail, currentPoint) <= CURRENT_ANCHOR_THRESHOLD_M
      ? [...orderedHistory.slice(0, -1), currentPoint]
      : [...orderedHistory, currentPoint];

  const finalPolyline = dedupePolyline(merged, 1e-3);
  const targetSamples = Math.round(clamp(finalPolyline.length * 4, 12, 48));

  return resamplePolyline(finalPolyline, targetSamples);
}

function formatAgentLabel(label: string) {
  if (!label) return "Agent";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function samplePolyline(points: Point[], t: number) {
  if (points.length === 0) return null;
  if (points.length === 1) return points[0];

  const position = clamp(t, 0, 1) * (points.length - 1);
  const index = Math.floor(position);
  const nextIndex = Math.min(index + 1, points.length - 1);
  const alpha = position - index;

  return {
    x: lerp(points[index].x, points[nextIndex].x, alpha),
    y: lerp(points[index].y, points[nextIndex].y, alpha),
  };
}

function sampleHeadingFromPolyline(points: Point[], t: number) {
  if (points.length < 2) return null;

  const position = clamp(t, 0, 0.999999) * (points.length - 1);
  const index = Math.floor(position);
  const nextIndex = Math.min(index + 1, points.length - 1);

  const dx = points[nextIndex].x - points[index].x;
  const dy = points[nextIndex].y - points[index].y;

  if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.abs(dx) + Math.abs(dy) < 1e-6) {
    return null;
  }

  return Math.atan2(dy, dx);
}

function pointInsideAgent(point: Point, agent: ParsedAgent) {
  const dx = point.x - agent.x;
  const dy = point.y - agent.y;

  const c = Math.cos(agent.heading);
  const s = Math.sin(agent.heading);

  const localX = dx * c + dy * s;
  const localY = -dx * s + dy * c;

  return Math.abs(localX) <= agent.length / 2 && Math.abs(localY) <= agent.width / 2;
}

function drawRoundedRectPath(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));

  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function drawPolyline(
  context: CanvasRenderingContext2D,
  points: Point[],
  camera: Camera,
) {
  if (points.length < 2) return;

  context.beginPath();
  points.forEach((point, index) => {
    const screen = worldToScreen(point.x, point.y, camera);
    if (index === 0) {
      context.moveTo(screen.x, screen.y);
    } else {
      context.lineTo(screen.x, screen.y);
    }
  });
  context.stroke();
}

function drawAgentShape(
  context: CanvasRenderingContext2D,
  camera: Camera,
  palette: Palette,
  options: {
    x: number;
    y: number;
    heading: number;
    length: number;
    width: number;
    bodyColor: string;
    isEgo?: boolean;
    ghost?: boolean;
    highlight?: boolean;
  },
) {
  const {
    x,
    y,
    heading,
    length,
    width,
    bodyColor,
    isEgo = false,
    ghost = false,
    highlight = false,
  } = options;

  const center = worldToScreen(x, y, camera);
  const pxLength = Math.max(length * camera.scale, isEgo ? 18 : 10);
  const pxWidth = Math.max(width * camera.scale, isEgo ? 9 : 6);
  const radius = Math.min(pxWidth * 0.35, 8);

  context.save();
  context.translate(center.x, center.y);
  context.rotate(-heading);

  if (!ghost) {
    context.save();
    context.translate(1.5, 2.5);
    drawRoundedRectPath(context, -pxLength / 2, -pxWidth / 2, pxLength, pxWidth, radius);
    context.fillStyle = palette.agentShadow;
    context.fill();
    context.restore();
  }

  context.globalAlpha = ghost ? 0.42 : 1;

  drawRoundedRectPath(context, -pxLength / 2, -pxWidth / 2, pxLength, pxWidth, radius);
  context.fillStyle = bodyColor;
  context.fill();

  drawRoundedRectPath(context, -pxLength / 2, -pxWidth / 2, pxLength, pxWidth, radius);
  context.lineWidth = highlight ? 2.4 : isEgo ? 2.2 : 1.4;
  context.strokeStyle = highlight ? palette.hover : palette.outline;
  context.stroke();

  drawRoundedRectPath(
    context,
    -pxLength * 0.12,
    -pxWidth * 0.28,
    pxLength * 0.42,
    pxWidth * 0.56,
    Math.min(pxWidth * 0.24, 5),
  );
  context.fillStyle = palette.roofLight;
  context.fill();

  context.beginPath();
  context.moveTo(pxLength * 0.16, 0);
  context.lineTo(pxLength * 0.46, -pxWidth * 0.18);
  context.lineTo(pxLength * 0.46, pxWidth * 0.18);
  context.closePath();
  context.fillStyle = palette.heading;
  context.fill();

  if (highlight) {
    context.beginPath();
    context.ellipse(0, 0, pxLength * 0.75, pxWidth * 0.92, 0, 0, Math.PI * 2);
    context.lineWidth = 1.5;
    context.strokeStyle = palette.hover;
    context.stroke();
  }

  if (isEgo && !ghost) {
    context.beginPath();
    context.moveTo(-pxLength * 0.18, -pxWidth / 2);
    context.lineTo(-pxLength * 0.18, pxWidth / 2);
    context.lineWidth = 1.5;
    context.strokeStyle = palette.heading;
    context.stroke();
  }

  context.restore();
}

export default function VectorSceneViewerClient({ indexUrl }: Props) {
  const [indexData, setIndexData] = useState<SceneIndex | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [scene, setScene] = useState<LoadedScene | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [themeVersion, setThemeVersion] = useState(0);

  const [showRoute, setShowRoute] = useState(true);
  const [showTiles, setShowTiles] = useState(true);
  const [showConnections, setShowConnections] = useState(false);
  const [showMotion, setShowMotion] = useState(true);
  const [animateMotion, setAnimateMotion] = useState(false);
  const [motionProgress, setMotionProgress] = useState(0.35);

  const [camera, setCamera] = useState<Camera | null>(null);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const dragRef = useRef({
    active: false,
    pointerId: -1,
    lastClientX: 0,
    lastClientY: 0,
  });

  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    cameraRef.current = camera;
  }, [camera]);

  useEffect(() => {
    let cancelled = false;

    async function loadIndex() {
      try {
        const absoluteIndexUrl = new URL(indexUrl, window.location.href).toString();
        const data = await fetchJson<SceneIndex>(absoluteIndexUrl);

        if (cancelled) return;

        setIndexData(data);
        if (data.scenes.length > 0) {
          setSelectedId((current) => current || data.scenes[0].id);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error ? loadError.message : "Failed to load scene index.",
          );
        }
      }
    }

    void loadIndex();

    return () => {
      cancelled = true;
    };
  }, [indexUrl]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;

    const updateSize = () => {
      setViewportSize({
        width: node.clientWidth,
        height: node.clientHeight,
      });
    };

    updateSize();

    const observer = new ResizeObserver(updateSize);
    observer.observe(node);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleThemeChange = () => setThemeVersion((value) => value + 1);

    const observer = new MutationObserver(handleThemeChange);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", handleThemeChange);

    return () => {
      observer.disconnect();
      media.removeEventListener("change", handleThemeChange);
    };
  }, []);

  useEffect(() => {
    const currentIndexData = indexData;
    if (!currentIndexData || !selectedId) return;

    const selectedEntry = currentIndexData.scenes.find((item) => item.id === selectedId);
    if (!selectedEntry) return;

    const absoluteIndexUrl = new URL(indexUrl, window.location.href).toString();
    let cancelled = false;

    async function loadScene(entry: IndexSceneEntry) {
      setLoading(true);
      setError("");
      setScene(null);
      setCamera(null);
      setHoverState(null);

      try {
        const sceneUrl = new URL(entry.path, absoluteIndexUrl).toString();
        const meta = await fetchJson<SceneMeta>(sceneUrl);

        const [
          agentStates,
          agentTypes,
          lanes,
          route,
          tileCorners,
          laneConnSrc,
          laneConnDst,
          laneConnType,
          agentMotion,
        ] = (await Promise.all([
          fetchTypedArray(
            new URL(meta.buffers.agent_states.file, sceneUrl).toString(),
            meta.buffers.agent_states.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.agent_types.file, sceneUrl).toString(),
            meta.buffers.agent_types.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.lanes.file, sceneUrl).toString(),
            meta.buffers.lanes.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.route.file, sceneUrl).toString(),
            meta.buffers.route.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.tile_corners.file, sceneUrl).toString(),
            meta.buffers.tile_corners.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.lane_connections.src.file, sceneUrl).toString(),
            meta.buffers.lane_connections.src.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.lane_connections.dst.file, sceneUrl).toString(),
            meta.buffers.lane_connections.dst.dtype,
          ),
          fetchTypedArray(
            new URL(meta.buffers.lane_connections.type.file, sceneUrl).toString(),
            meta.buffers.lane_connections.type.dtype,
          ),
          meta.buffers.agent_motion
            ? fetchTypedArray(
                new URL(meta.buffers.agent_motion.file, sceneUrl).toString(),
                meta.buffers.agent_motion.dtype,
              )
            : Promise.resolve(undefined),
        ])) as [
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

        if (cancelled) return;

        setScene({
          meta,
          agentStates,
          agentTypes,
          agentMotion,
          lanes,
          route,
          tileCorners,
          laneConnSrc,
          laneConnDst,
          laneConnType,
        });
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load scene.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadScene(selectedEntry);

    return () => {
      cancelled = true;
    };
  }, [indexData, selectedId, indexUrl]);

  const selectedIndexEntry = useMemo(
    () => indexData?.scenes.find((item) => item.id === selectedId) ?? null,
    [indexData, selectedId],
  );

  const parsedScene = useMemo<ParsedScene | null>(() => {
    if (!scene) return null;

    const meta = scene.meta;

    const pointsPerLane = meta.buffers.lanes.shape[1] ?? meta.counts.num_points_per_lane;
    const numLanes = Math.min(
      meta.counts.num_lanes,
      Math.floor(scene.lanes.length / Math.max(pointsPerLane * 2, 1)),
    );

    const routePointCount = Math.min(meta.counts.route_points, Math.floor(scene.route.length / 2));
    const tileCount = Math.min(meta.counts.tiles, Math.floor(scene.tileCorners.length / 8));

    const stateStride = lastShapeDim(
      meta.buffers.agent_states.shape,
      Math.max(1, Math.floor(scene.agentStates.length / Math.max(meta.counts.num_agents, 1))),
    );

    const motionStride =
      scene.agentMotion && meta.buffers.agent_motion
        ? lastShapeDim(
            meta.buffers.agent_motion.shape,
            Math.max(0, Math.floor(scene.agentMotion.length / Math.max(meta.counts.num_agents, 1))),
          )
        : 0;

    const numAgents = Math.min(
      meta.counts.num_agents,
      scene.agentTypes.length,
      Math.floor(scene.agentStates.length / Math.max(stateStride, 1)),
    );

    const typeLabels = meta.buffers.agent_types.labels ?? ["vehicle", "pedestrian", "cyclist"];

    const lanes: Point[][] = Array.from({ length: numLanes }, () => []);

    for (let laneIndex = 0; laneIndex < numLanes; laneIndex += 1) {
      const points: Point[] = [];

      for (let pointIndex = 0; pointIndex < pointsPerLane; pointIndex += 1) {
        const base = (laneIndex * pointsPerLane + pointIndex) * 2;
        const x = scene.lanes[base];
        const y = scene.lanes[base + 1];

        if (isFiniteNumber(x) && isFiniteNumber(y)) {
          points.push({ x, y });
        }
      }

      lanes[laneIndex] = points;
    }

    const route: Point[] = [];
    for (let i = 0; i < routePointCount; i += 1) {
      const base = i * 2;
      const x = scene.route[base];
      const y = scene.route[base + 1];
      if (isFiniteNumber(x) && isFiniteNumber(y)) {
        route.push({ x, y });
      }
    }

    const tiles: Point[][] = [];
    for (let tile = 0; tile < tileCount; tile += 1) {
      const corners: Point[] = [];
      for (let corner = 0; corner < 4; corner += 1) {
        const base = tile * 8 + corner * 2;
        const x = scene.tileCorners[base];
        const y = scene.tileCorners[base + 1];
        if (isFiniteNumber(x) && isFiniteNumber(y)) {
          corners.push({ x, y });
        }
      }
      tiles.push(corners);
    }

    const agents: ParsedAgent[] = [];
    for (let agentIndex = 0; agentIndex < numAgents; agentIndex += 1) {
      const stateBase = agentIndex * stateStride;
      const x = scene.agentStates[stateBase + STATE_LAYOUT.x];
      const y = scene.agentStates[stateBase + STATE_LAYOUT.y];
      const headingValue = scene.agentStates[stateBase + STATE_LAYOUT.heading];
      const heading = isFiniteNumber(headingValue) ? headingValue : 0;

      if (!isFiniteNumber(x) || !isFiniteNumber(y)) continue;

      const typeIndex = scene.agentTypes[agentIndex] ?? 0;
      const label = typeLabels[typeIndex] ?? "vehicle";

      const rawLength =
        STATE_LAYOUT.length < stateStride
          ? scene.agentStates[stateBase + STATE_LAYOUT.length]
          : undefined;

      const rawWidth =
        STATE_LAYOUT.width < stateStride
          ? scene.agentStates[stateBase + STATE_LAYOUT.width]
          : undefined;

      const { length, width } = resolveAgentSize(label, rawLength, rawWidth);

      const motion =
        scene.agentMotion && motionStride >= 2
          ? buildHistoryMotionPolyline(
              x,
              y,
              heading,
              scene.agentMotion,
              agentIndex * motionStride,
              motionStride,
            )
          : [{ x, y }];

      agents.push({
        id: agentIndex,
        x,
        y,
        heading,
        length,
        width,
        label,
        isEgo: agentIndex === meta.ego_index,
        motion,
      });
    }

    const enumMap = meta.buffers.lane_connections.enum;
    const connectionCount = Math.min(
      scene.laneConnSrc.length,
      scene.laneConnDst.length,
      scene.laneConnType.length,
    );

    const connections: ParsedConnection[] = [];
    for (let index = 0; index < connectionCount; index += 1) {
      const srcLane = scene.laneConnSrc[index];
      const dstLane = scene.laneConnDst[index];
      const typeValue = scene.laneConnType[index];

      if (srcLane >= lanes.length || dstLane >= lanes.length) continue;
      const srcPoints = lanes[srcLane];
      const dstPoints = lanes[dstLane];
      if (srcPoints.length === 0 || dstPoints.length === 0) continue;

      const kind: ParsedConnectionKind =
        typeValue === enumMap.succ
          ? "succ"
          : typeValue === enumMap.left
            ? "left"
            : typeValue === enumMap.right
              ? "right"
              : "other";

      connections.push({
        srcTail: srcPoints[srcPoints.length - 1],
        dstHead: dstPoints[0],
        kind,
      });
    }

    return {
      meta,
      agents,
      lanes,
      route,
      tiles,
      connections,
    };
  }, [scene]);

  useEffect(() => {
    if (!scene || viewportSize.width === 0 || viewportSize.height === 0) return;
    setCamera(createFitCamera(scene.meta.bbox_xyxy, viewportSize.width, viewportSize.height, 56));
    setHoverState(null);
    setMotionProgress(0.35);
  }, [scene?.meta.scene_id, viewportSize.width, viewportSize.height]);

  useEffect(() => {
    if (!animateMotion) return;

    let raf = 0;
    let lastTimestamp: number | null = null;

    const tick = (timestamp: number) => {
      if (lastTimestamp === null) {
        lastTimestamp = timestamp;
      }

      const delta = (timestamp - lastTimestamp) / 1000;
      lastTimestamp = timestamp;

      setMotionProgress((current) => {
        const next = current + delta / MOTION_LOOP_SECONDS;
        return next >= 1 ? next - Math.floor(next) : next;
      });

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(raf);
  }, [animateMotion]);

  const fitToScene = () => {
    if (!scene || viewportSize.width === 0 || viewportSize.height === 0) return;
    setCamera(createFitCamera(scene.meta.bbox_xyxy, viewportSize.width, viewportSize.height, 56));
    setHoverState(null);
  };

  const centerOnEgo = () => {
    if (!parsedScene || viewportSize.width === 0 || viewportSize.height === 0) return;

    const ego = parsedScene.agents.find((agent) => agent.isEgo) ?? parsedScene.agents[0];
    if (!ego) return;

    setCamera((current) => {
      if (!current) return current;
      return {
        ...current,
        tx: viewportSize.width / 2 - ego.x * current.scale,
        ty: viewportSize.height / 2 + ego.y * current.scale,
      };
    });
  };

  const zoomAroundPoint = (factor: number, anchorX: number, anchorY: number) => {
    setCamera((current) => {
      if (!current) return current;

      const worldPoint = screenToWorld(anchorX, anchorY, current);
      const nextScale = clamp(
        current.scale * factor,
        current.fitScale * 0.6,
        current.fitScale * 18,
      );

      return {
        ...current,
        scale: nextScale,
        tx: anchorX - worldPoint.x * nextScale,
        ty: anchorY + worldPoint.y * nextScale,
      };
    });
  };

  const updateHoverFromScreen = (screenX: number, screenY: number) => {
    if (!parsedScene || !cameraRef.current) {
      setHoverState(null);
      return;
    }

    const cameraValue = cameraRef.current;
    const worldPoint = screenToWorld(screenX, screenY, cameraValue);

    let found: ParsedAgent | null = null;

    for (let index = parsedScene.agents.length - 1; index >= 0; index -= 1) {
      const agent = parsedScene.agents[index];
      if (pointInsideAgent(worldPoint, agent)) {
        found = agent;
        break;
      }
    }

    if (!found) {
      let bestDistance = 16;

      parsedScene.agents.forEach((agent) => {
        const screen = worldToScreen(agent.x, agent.y, cameraValue);
        const distance = Math.hypot(screen.x - screenX, screen.y - screenY);
        if (distance < bestDistance) {
          bestDistance = distance;
          found = agent;
        }
      });
    }

    setHoverState(
      found
        ? {
            agent: found,
            screenX,
            screenY,
          }
        : null,
    );
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;

    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      active: true,
      pointerId: event.pointerId,
      lastClientX: event.clientX,
      lastClientY: event.clientY,
    };
    setIsDragging(true);
    setHoverState(null);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return;

    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;

    if (dragRef.current.active) {
      const dx = event.clientX - dragRef.current.lastClientX;
      const dy = event.clientY - dragRef.current.lastClientY;

      dragRef.current.lastClientX = event.clientX;
      dragRef.current.lastClientY = event.clientY;

      setCamera((current) =>
        current
          ? {
              ...current,
              tx: current.tx + dx,
              ty: current.ty + dy,
            }
          : current,
      );
      return;
    }

    updateHoverFromScreen(localX, localY);
  };

  const handlePointerUpOrCancel = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current.active && dragRef.current.pointerId === event.pointerId) {
      try {
        event.currentTarget.releasePointerCapture(event.pointerId);
      } catch {
        // ignore
      }
    }

    dragRef.current.active = false;
    dragRef.current.pointerId = -1;
    setIsDragging(false);

    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return;

    updateHoverFromScreen(event.clientX - rect.left, event.clientY - rect.top);
  };

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();

    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return;

    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const factor = event.deltaY < 0 ? 1.12 : 0.89;

    zoomAroundPoint(factor, localX, localY);
  };

  useEffect(() => {
    if (!parsedScene || !camera || !canvasRef.current) return;
    if (viewportSize.width === 0 || viewportSize.height === 0) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    const devicePixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(viewportSize.width * devicePixelRatio);
    canvas.height = Math.floor(viewportSize.height * devicePixelRatio);
    canvas.style.width = `${viewportSize.width}px`;
    canvas.style.height = `${viewportSize.height}px`;

    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    context.clearRect(0, 0, viewportSize.width, viewportSize.height);
    context.imageSmoothingEnabled = true;

    const dark = isDarkTheme();

    const palette: Palette = dark
      ? {
          background: "#09090b",
          roadFill: "rgba(148,163,184,0.18)",
          laneCenter: "rgba(228,228,231,0.52)",
          routeGlow: "rgba(56,189,248,0.20)",
          route: "#38bdf8",
          tileFill: "rgba(59,130,246,0.08)",
          tileStroke: "rgba(96,165,250,0.28)",
          successor: "rgba(34,197,94,0.72)",
          lateral: "rgba(168,85,247,0.72)",
          vehicle: "#e4e4e7",
          pedestrian: "#4ade80",
          cyclist: "#facc15",
          ego: "#fb923c",
          roofLight: "rgba(9,9,11,0.24)",
          outline: "rgba(9,9,11,0.96)",
          heading: "rgba(9,9,11,0.92)",
          hover: "#60a5fa",
          motionVehicle: "rgba(56,189,248,0.46)",
          motionPedestrian: "rgba(74,222,128,0.40)",
          motionCyclist: "rgba(250,204,21,0.40)",
          agentShadow: "rgba(0,0,0,0.28)",
        }
      : {
          background: "#ffffff",
          roadFill: "rgba(148,163,184,0.28)",
          laneCenter: "rgba(51,65,85,0.62)",
          routeGlow: "rgba(14,165,233,0.18)",
          route: "#0284c7",
          tileFill: "rgba(37,99,235,0.05)",
          tileStroke: "rgba(37,99,235,0.18)",
          successor: "rgba(34,197,94,0.72)",
          lateral: "rgba(168,85,247,0.68)",
          vehicle: "#334155",
          pedestrian: "#16a34a",
          cyclist: "#ca8a04",
          ego: "#ea580c",
          roofLight: "rgba(255,255,255,0.28)",
          outline: "rgba(255,255,255,0.96)",
          heading: "rgba(255,255,255,0.92)",
          hover: "#2563eb",
          motionVehicle: "rgba(2,132,199,0.42)",
          motionPedestrian: "rgba(22,163,74,0.38)",
          motionCyclist: "rgba(202,138,4,0.38)",
          agentShadow: "rgba(15,23,42,0.16)",
        };

    const laneWidth = isFiniteNumber(parsedScene.meta.lane_width_m)
      ? parsedScene.meta.lane_width_m
      : 4.2;

    const roadBandWidth = clamp(laneWidth * camera.scale * 0.6, 1.8, 24);
    const laneCenterWidth = clamp(roadBandWidth * 0.14, 1.1, 2.8);

    context.fillStyle = palette.background;
    context.fillRect(0, 0, viewportSize.width, viewportSize.height);

    if (showTiles) {
      context.save();
      context.fillStyle = palette.tileFill;
      context.strokeStyle = palette.tileStroke;
      context.lineWidth = 1;

      parsedScene.tiles.forEach((tile) => {
        if (tile.length < 4) return;

        context.beginPath();
        tile.forEach((corner, index) => {
          const screen = worldToScreen(corner.x, corner.y, camera);
          if (index === 0) {
            context.moveTo(screen.x, screen.y);
          } else {
            context.lineTo(screen.x, screen.y);
          }
        });
        context.closePath();
        context.fill();
        context.stroke();
      });

      context.restore();
    }

    context.save();
    context.strokeStyle = palette.roadFill;
    context.lineWidth = roadBandWidth;
    context.lineJoin = "round";
    context.lineCap = "round";

    parsedScene.lanes.forEach((lanePoints) => {
      if (lanePoints.length < 2) return;
      drawPolyline(context, lanePoints, camera);
    });

    context.restore();

    context.save();
    context.strokeStyle = palette.laneCenter;
    context.lineWidth = laneCenterWidth;
    context.lineJoin = "round";
    context.lineCap = "round";

    parsedScene.lanes.forEach((lanePoints) => {
      if (lanePoints.length < 2) return;
      drawPolyline(context, lanePoints, camera);
    });

    context.restore();

    if (showConnections) {
      context.save();
      context.lineWidth = 1.2;
      context.setLineDash([6, 4]);
      context.lineCap = "round";

      parsedScene.connections.forEach((connection) => {
        if (connection.kind === "other") return;

        context.strokeStyle =
          connection.kind === "succ" ? palette.successor : palette.lateral;

        context.beginPath();
        const p0 = worldToScreen(connection.srcTail.x, connection.srcTail.y, camera);
        const p1 = worldToScreen(connection.dstHead.x, connection.dstHead.y, camera);
        context.moveTo(p0.x, p0.y);
        context.lineTo(p1.x, p1.y);
        context.stroke();
      });

      context.restore();
    }

    if (showRoute && parsedScene.route.length > 1) {
      context.save();
      context.strokeStyle = palette.routeGlow;
      context.lineWidth = clamp(roadBandWidth * 0.9, 4, 18);
      context.lineJoin = "round";
      context.lineCap = "round";
      drawPolyline(context, parsedScene.route, camera);
      context.restore();

      context.save();
      context.strokeStyle = palette.route;
      context.lineWidth = clamp(roadBandWidth * 0.3, 2.2, 6.2);
      context.lineJoin = "round";
      context.lineCap = "round";
      drawPolyline(context, parsedScene.route, camera);
      context.restore();
    }

    const hoveredAgentId = hoverState?.agent.id ?? -1;

    if (showMotion) {
      parsedScene.agents.forEach((agent) => {
        if (agent.motion.length < 2) return;

        const motionColor =
          agent.label === "pedestrian"
            ? palette.motionPedestrian
            : agent.label === "cyclist"
              ? palette.motionCyclist
              : palette.motionVehicle;

        context.save();
        context.strokeStyle = motionColor;
        context.lineWidth = agent.isEgo ? 2.4 : 1.5;
        context.lineCap = "round";
        context.lineJoin = "round";
        drawPolyline(context, agent.motion, camera);
        context.restore();

        const ghostPoint = samplePolyline(agent.motion, motionProgress);
        const ghostHeading = sampleHeadingFromPolyline(agent.motion, motionProgress) ?? agent.heading;

        if (ghostPoint) {
          const bodyColor =
            agent.isEgo
              ? palette.ego
              : agent.label === "pedestrian"
                ? palette.pedestrian
                : agent.label === "cyclist"
                  ? palette.cyclist
                  : palette.vehicle;

          drawAgentShape(context, camera, palette, {
            x: ghostPoint.x,
            y: ghostPoint.y,
            heading: ghostHeading,
            length: agent.length,
            width: agent.width,
            bodyColor,
            isEgo: agent.isEgo,
            ghost: true,
            highlight: false,
          });
        }
      });

      if (hoverState?.agent.motion && hoverState.agent.motion.length > 1) {
        context.save();
        context.strokeStyle = palette.hover;
        context.lineWidth = 2.4;
        context.lineJoin = "round";
        context.lineCap = "round";
        drawPolyline(context, hoverState.agent.motion, camera);
        context.restore();
      }
    }

    const agentsForDraw = [...parsedScene.agents].sort((a, b) => {
      if (a.id === hoveredAgentId) return 1;
      if (b.id === hoveredAgentId) return -1;
      if (a.isEgo === b.isEgo) return 0;
      return a.isEgo ? 1 : -1;
    });

    agentsForDraw.forEach((agent) => {
      const bodyColor =
        agent.isEgo
          ? palette.ego
          : agent.label === "pedestrian"
            ? palette.pedestrian
            : agent.label === "cyclist"
              ? palette.cyclist
              : palette.vehicle;

      drawAgentShape(context, camera, palette, {
        x: agent.x,
        y: agent.y,
        heading: agent.heading,
        length: agent.length,
        width: agent.width,
        bodyColor,
        isEgo: agent.isEgo,
        ghost: false,
        highlight: agent.id === hoveredAgentId,
      });
    });
  }, [
    parsedScene,
    camera,
    viewportSize,
    showRoute,
    showTiles,
    showConnections,
    showMotion,
    hoverState,
    motionProgress,
    themeVersion,
  ]);

  const chipClass =
    "rounded-full bg-zinc-100 px-3 py-1 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300";

  const secondaryButtonClass =
    "rounded-full border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800";

  const relativeZoom = camera ? camera.scale / camera.fitScale : 1;

  const tooltipLeft =
    hoverState && viewportSize.width > 0
      ? Math.max(12, Math.min(hoverState.screenX + 16, viewportSize.width - 236))
      : 0;

  const tooltipTop =
    hoverState && viewportSize.height > 0
      ? Math.max(12, Math.min(hoverState.screenY - 12, viewportSize.height - 148))
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
                onChange={(event) => setSelectedId(event.target.value)}
                className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
              >
                {indexData?.scenes.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.id}
                  </option>
                ))}
              </select>
            </label>

            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={fitToScene} className="rounded-full bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white">
                Fit view
              </button>
              <button type="button" onClick={centerOnEgo} className={secondaryButtonClass}>
                Center ego
              </button>
              <button
                type="button"
                onClick={() => zoomAroundPoint(1.2, viewportSize.width / 2, viewportSize.height / 2)}
                className={secondaryButtonClass}
              >
                Zoom in
              </button>
              <button
                type="button"
                onClick={() => zoomAroundPoint(1 / 1.2, viewportSize.width / 2, viewportSize.height / 2)}
                className={secondaryButtonClass}
              >
                Zoom out
              </button>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <div>
                <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  Inspection controls
                </h3>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Interactive view of the exported vector graph
                </p>
              </div>

              <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
                Pan, zoom, and inspect individual agents directly in the exported scene.
              </p>

              <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
                Agents are rendered as oriented boxes to preserve geometry, heading, and local interaction cues.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Layers
              </h3>

              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={showRoute}
                    onChange={(event) => setShowRoute(event.target.checked)}
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                  />
                  Route
                </label>

                <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={showTiles}
                    onChange={(event) => setShowTiles(event.target.checked)}
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                  />
                  Tile bounds
                </label>

                <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={showConnections}
                    onChange={(event) => setShowConnections(event.target.checked)}
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                  />
                  Lane links
                </label>

                <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={showMotion}
                    onChange={(event) => {
                      const checked = event.target.checked;
                      setShowMotion(checked);
                      if (!checked) {
                        setAnimateMotion(false);
                      }
                    }}
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                  />
                  Motion trail
                </label>
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
                onChange={(event) => {
                  setAnimateMotion(false);
                  setMotionProgress(Number(event.target.value) / 1000);
                }}
                className="w-full accent-blue-600"
              />

              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={!showMotion}
                  onClick={() => setAnimateMotion((current) => !current)}
                  className="rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-400"
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
                  className="rounded-full border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
                >
                  Reset trail
                </button>
              </div>

              <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
                Qualitative trail preview reconstructed from <code>agent_motion</code>.
              </p>
            </div>
          </div>
        </div>

        {selectedIndexEntry && (
          <div className="mt-5 flex flex-wrap gap-2">
            <span className={chipClass}>scene {selectedIndexEntry.id}</span>
            <span className={chipClass}>{scene?.meta.dataset ?? "dataset"}</span>
            <span className={chipClass}>{selectedIndexEntry.num_lanes} lanes</span>
            <span className={chipClass}>{selectedIndexEntry.num_agents} agents</span>
            <span className={chipClass}>{selectedIndexEntry.route_points} route points</span>
            <span className={chipClass}>{relativeZoom.toFixed(2)}× zoom</span>
            <span
              className={
                selectedIndexEntry.route_completed
                  ? "rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 dark:bg-green-500/10 dark:text-green-300"
                  : chipClass
              }
            >
              {selectedIndexEntry.route_completed ? "route completed" : "route not completed"}
            </span>
          </div>
        )}
      </section>

      <div
        ref={stageRef}
        aria-label="Interactive vector scene viewer"
        className={`relative w-full overflow-hidden rounded-3xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950 ${
          isDragging ? "cursor-grabbing" : "cursor-grab"
        } select-none touch-none`}
        style={{ height: "clamp(30rem, 74vh, 46rem)" }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUpOrCancel}
        onPointerCancel={handlePointerUpOrCancel}
        onPointerLeave={() => {
          if (!dragRef.current.active) {
            setHoverState(null);
          }
        }}
        onWheel={handleWheel}
        onDoubleClick={(event) => {
          event.preventDefault();
          fitToScene();
        }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        <div className="pointer-events-none absolute left-4 top-4 z-10 flex flex-wrap gap-2 text-xs text-zinc-700 dark:text-zinc-300">
          <span className="rounded-full border border-zinc-200 bg-white/90 px-3 py-1 backdrop-blur dark:border-zinc-700 dark:bg-zinc-900/90">
            Interactive canvas
          </span>
          <span className="rounded-full border border-zinc-200 bg-white/90 px-3 py-1 backdrop-blur dark:border-zinc-700 dark:bg-zinc-900/90">
            Double-click to fit
          </span>
        </div>

        {(loading || !scene || !camera) && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 text-sm font-medium text-zinc-700 backdrop-blur dark:bg-zinc-950/80 dark:text-zinc-300">
            Loading vector scene…
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/85 px-6 text-center text-sm font-medium text-red-700 backdrop-blur dark:bg-zinc-950/85 dark:text-red-300">
            {error}
          </div>
        )}

        {hoverState && (
          <div
            className="pointer-events-none absolute z-10 w-56 rounded-2xl border border-zinc-200 bg-white/95 p-3 text-xs shadow-lg backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/95"
            style={{
              left: tooltipLeft,
              top: tooltipTop,
            }}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="font-semibold text-zinc-900 dark:text-zinc-100">
                agent {hoverState.agent.id}
              </div>
              <div className="rounded-full bg-zinc-100 px-2 py-0.5 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                {hoverState.agent.isEgo
                  ? "Ego"
                  : formatAgentLabel(hoverState.agent.label)}
              </div>
            </div>

            <div className="mt-3 space-y-1.5 text-zinc-600 dark:text-zinc-400">
              <div>
                position ({hoverState.agent.x.toFixed(1)}, {hoverState.agent.y.toFixed(1)}) m
              </div>
              <div>heading {toDegrees(hoverState.agent.heading).toFixed(1)}°</div>
              <div>
                bbox {hoverState.agent.length.toFixed(1)} × {hoverState.agent.width.toFixed(1)} m
              </div>
              <div>{Math.max(0, hoverState.agent.motion.length - 1)} trail samples</div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-4 text-xs text-zinc-600 dark:text-zinc-400">
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-zinc-400"></span>
          Lane band
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-sky-500"></span>
          Route
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-blue-400"></span>
          Motion trail
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-orange-500"></span>
          Ego
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-slate-500"></span>
          Vehicle
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-green-500"></span>
          Pedestrian
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-amber-500"></span>
          Cyclist
        </span>
      </div>

      <div className="rounded-3xl border border-zinc-200 bg-zinc-50 p-4 text-sm leading-6 text-zinc-600 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
        <strong className="text-zinc-900 dark:text-zinc-100">Note.</strong>{" "}
        Exported scene inspection only: one scene snapshot plus a qualitative{" "}
        <code>agent_motion</code> trail, not a full per-step replay.
      </div>
    </div>
  );
}