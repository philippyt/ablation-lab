import { telemetry } from "../state";

interface SparkProps {
  title: string;
  unit: string;
  values: number[];
  color: string;
  fill: string;
}

function SparkCard({ title, unit, values, color, fill }: SparkProps) {
  const W = 220;
  const H = 56;
  const PAD_X = 4;
  const PAD_Y = 6;
  const hasData = values.length > 0;
  const last = hasData ? values[values.length - 1] : 0;
  const peak = hasData ? Math.max(...values) : 0;
  const minV = hasData ? Math.min(...values) : 0;
  const maxV = Math.max(peak, minV + 0.01);

  let linePts = "";
  let areaPts = "";
  if (values.length >= 2) {
    const coords = values.map((v, i) => {
      const x = PAD_X + (i / (values.length - 1)) * (W - PAD_X * 2);
      const y = PAD_Y + (1 - (v - minV) / (maxV - minV)) * (H - PAD_Y * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    linePts = coords.join(" ");
    areaPts = `${PAD_X},${H - PAD_Y} ${linePts} ${W - PAD_X},${H - PAD_Y}`;
  }

  return (
    <div class="spark-card">
      <div class="spark-head">
        <span class="spark-title">{title}</span>
        <span class="spark-val">
          {hasData ? last.toFixed(2) : "0.00"}
          <span class="spark-unit">{unit}</span>
        </span>
      </div>
      <svg class="spark-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {areaPts && <polygon points={areaPts} fill={fill} />}
        {linePts ? (
          <polyline points={linePts} fill="none" stroke={color} stroke-width="1.8" stroke-linejoin="round" />
        ) : (
          <line x1={PAD_X} y1={H / 2} x2={W - PAD_X} y2={H / 2} stroke="#d4d7de" stroke-width="1.2" stroke-dasharray="3 3" />
        )}
      </svg>
      <div class="spark-foot">
        <span>Peak {hasData ? peak.toFixed(2) : "0.00"}</span>
        <span>{values.length} steps</span>
      </div>
    </div>
  );
}

export function LiveTelemetry() {
  const pts = telemetry.value;
  const logitMax = pts.map((p) => p.lm);
  const entropy = pts.map((p) => p.ent);
  const residNorm = pts.map((p) => p.rn);

  return (
    <div class="telemetry-strip">
      <SparkCard
        title="Max attn logit"
        unit=""
        values={logitMax}
        color="#c2410c"
        fill="rgba(194, 65, 12, 0.12)"
      />
      <SparkCard
        title="Next token entropy"
        unit=" nats"
        values={entropy}
        color="#1d4ed8"
        fill="rgba(29, 78, 216, 0.12)"
      />
      <SparkCard
        title="Final residual norm"
        unit=""
        values={residNorm}
        color="#0f766e"
        fill="rgba(15, 118, 110, 0.12)"
      />
    </div>
  );
}
